import argparse
import ctypes
import math
import shutil
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np
import psutil
import pystray
import tkinter as tk
from PIL import Image, ImageDraw
from pystray import MenuItem as TrayItem
from tkinter import filedialog, messagebox, ttk


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class AppConfig:
    input_fps: float
    output_fps: float
    monitor_index: int
    video_format: str
    max_width: int
    paused_dim_alpha: float
    default_speed_factor: float


@dataclass
class RecorderConfig:
    input_fps: float
    speed_factor: float
    output_fps: float
    monitor_index: int
    temp_output_path: Path
    default_save_name: str
    video_format: str
    max_width: int
    paused_dim_alpha: float


class TimelapseRecorder:
    def __init__(self, config: RecorderConfig):
        self.config = config
        self._lock = threading.Lock()
        self._paused = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._start_ts: Optional[float] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._writer_frame_size: Optional[Tuple[int, int]] = None
        self._last_live_frame: Optional[np.ndarray] = None
        self._frames_written = 0
        self._frames_captured = 0
        self._time_scale_accumulator = 0.0
        self._error: Optional[str] = None
        self._summary_tail_payload: Optional[dict] = None
        self._summary_tail_seconds = 5.0

    @property
    def interval_seconds(self) -> float:
        return 1.0 / self.config.input_fps

    def start(self) -> None:
        if self.is_running():
            return
        self._start_ts = time.monotonic()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def stop(self, summary_tail_payload: Optional[dict] = None) -> None:
        self._summary_tail_payload = summary_tail_payload
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._release_writer()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def toggle_pause(self) -> bool:
        with self._lock:
            self._paused = not self._paused
            return self._paused

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def elapsed_seconds(self) -> float:
        if self._start_ts is None:
            return 0.0
        return time.monotonic() - self._start_ts

    def frames_written(self) -> int:
        return self._frames_written

    def frames_captured(self) -> int:
        return self._frames_captured

    def get_error(self) -> Optional[str]:
        return self._error

    def temp_output_path(self) -> Path:
        return self.config.temp_output_path

    def move_temp_to(self, destination: Path) -> None:
        src = self.config.temp_output_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(destination))

    def _record_loop(self) -> None:
        try:
            next_tick = time.monotonic()
            with mss.mss() as sct:
                monitor = self._select_monitor(sct)
                while not self._stop_event.is_set():
                    now = time.monotonic()
                    if now < next_tick:
                        time.sleep(min(0.25, next_tick - now))
                        continue

                    paused = self.is_paused()
                    if paused and self._last_live_frame is not None:
                        frame = self._last_live_frame.copy()
                    else:
                        frame = self._grab_frame(sct, monitor)
                        self._last_live_frame = frame.copy()
                        self._frames_captured += 1

                    if paused:
                        frame = self._apply_pause_overlay(frame)
                    frame = self._apply_video_dashboard_overlay(frame, paused)

                    self._ensure_writer(frame)
                    assert self._writer is not None
                    repeats = self._compute_output_repeats()
                    for _ in range(repeats):
                        self._writer.write(frame)
                        self._frames_written += 1

                    next_tick += self.interval_seconds
                    if next_tick < time.monotonic() - self.interval_seconds:
                        next_tick = time.monotonic() + self.interval_seconds

                if self._writer is not None and self._summary_tail_payload is not None:
                    self._write_summary_tail(self._summary_tail_payload, self._summary_tail_seconds)
        except Exception as exc:
            self._error = str(exc)
            self._stop_event.set()

    def _select_monitor(self, sct: mss.mss) -> dict:
        monitors = sct.monitors
        if self.config.monitor_index < 1 or self.config.monitor_index >= len(monitors):
            raise ValueError(
                f"Invalid monitor index {self.config.monitor_index}. "
                f"Available: 1 to {len(monitors) - 1}"
            )
        return monitors[self.config.monitor_index]

    def _grab_frame(self, sct: mss.mss, monitor: dict) -> np.ndarray:
        shot = sct.grab(monitor)
        frame = np.array(shot, dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if self.config.max_width > 0 and frame.shape[1] > self.config.max_width:
            scale = self.config.max_width / frame.shape[1]
            new_h = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (self.config.max_width, new_h), interpolation=cv2.INTER_AREA)

        h, w = frame.shape[:2]
        if w % 2 == 1:
            frame = frame[:, :-1]
        if h % 2 == 1:
            frame = frame[:-1, :]
        return frame

    def _ensure_writer(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            return
        h, w = frame.shape[:2]
        self._writer_frame_size = (w, h)
        fourcc = self._codec_from_format(self.config.video_format)
        self.config.temp_output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.config.temp_output_path),
            fourcc,
            self.config.output_fps,
            (w, h),
        )
        if not self._writer.isOpened():
            raise RuntimeError(
                f"Could not open video writer for '{self.config.temp_output_path}'. "
                "Try --format avi if mp4 codec is unavailable."
            )

    def _release_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._writer_frame_size = None

    @staticmethod
    def _codec_from_format(video_format: str) -> int:
        if video_format == "avi":
            return cv2.VideoWriter_fourcc(*"XVID")
        return cv2.VideoWriter_fourcc(*"mp4v")

    def _compute_output_repeats(self) -> int:
        self._time_scale_accumulator += 1.0 / self.config.speed_factor
        repeats = int(self._time_scale_accumulator)
        self._time_scale_accumulator -= repeats
        return repeats

    def _apply_pause_overlay(self, frame: np.ndarray) -> np.ndarray:
        dimmed = cv2.addWeighted(
            frame,
            1.0 - self.config.paused_dim_alpha,
            np.zeros_like(frame),
            self.config.paused_dim_alpha,
            0.0,
        )
        h, w = dimmed.shape[:2]
        cx, cy = w // 2, h // 2
        bar_h = max(56, h // 5)
        bar_w = max(20, w // 45)
        gap = max(18, bar_w)

        left_x1 = cx - gap - bar_w
        left_x2 = cx - gap
        right_x1 = cx + gap
        right_x2 = cx + gap + bar_w
        y1 = cy - (bar_h // 2)
        y2 = cy + (bar_h // 2)

        cv2.rectangle(dimmed, (left_x1, y1), (left_x2, y2), (255, 255, 255), -1)
        cv2.rectangle(dimmed, (right_x1, y1), (right_x2, y2), (255, 255, 255), -1)
        cv2.putText(
            dimmed,
            "PAUSED",
            (cx - 95, min(h - 30, y2 + 70)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        return dimmed

    def _apply_video_dashboard_overlay(self, frame: np.ndarray, paused: bool) -> np.ndarray:
        bg = (42, 23, 15)
        bar_track = (59, 41, 30)
        bar_fill = (94, 197, 34)
        text_main = (240, 226, 226)
        text_sub = (252, 180, 165)

        h, w = frame.shape[:2]
        panel_h = max(88, h // 8)
        cv2.rectangle(frame, (0, 0), (w, panel_h), bg, -1)

        margin_x = 22
        bar_h = max(16, panel_h // 4)
        bar_y1 = panel_h // 2 - bar_h // 2 + 10
        bar_y2 = bar_y1 + bar_h
        bar_x1 = margin_x
        bar_x2 = w - margin_x

        elapsed_s = self.elapsed_seconds()
        scale_window_s = float(max(60, math.ceil(max(1.0, elapsed_s) / 60.0) * 60))
        fill_ratio = min(1.0, elapsed_s / scale_window_s)
        fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * fill_ratio)

        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), bar_track, -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x2, bar_y2), bar_fill, -1)
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 80, 80), 1)

        state = "PAUSED" if paused else "RECORDING"
        line1 = f"{state}  |  Elapsed {format_duration(elapsed_s)}"
        line2 = (
            f"Input {self.config.input_fps:.1f}fps  Output {self.config.output_fps:.1f}fps  "
            f"Speed {self.config.speed_factor:.2f}x  BarWindow {format_duration(scale_window_s)}"
        )
        cv2.putText(frame, line1, (margin_x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, text_main, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            line2,
            (margin_x, panel_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            text_sub,
            1,
            cv2.LINE_AA,
        )
        return frame

    def _write_summary_tail(self, payload: dict, seconds: float) -> None:
        if self._writer is None:
            return

        if self._writer_frame_size is not None:
            frame_w, frame_h = self._writer_frame_size
        else:
            frame_w = int(self._writer.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            frame_h = int(self._writer.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        duration = max(0.0, float(payload.get("elapsed_seconds", 0.0)))
        total_app_seconds = int(payload.get("total_app_seconds", 0))
        rows = list(payload.get("app_rows", []))
        paused_seconds = max(0.0, duration - float(total_app_seconds))

        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        frame[:, :] = (18, 18, 28)

        cv2.putText(
            frame,
            "Session Summary",
            (42, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            (240, 240, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Session Time: {format_duration(duration)}  |  Paused Time: {format_duration(paused_seconds)}",
            (42, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (196, 211, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Tracked App Focus Time: {format_duration(total_app_seconds)}",
            (42, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (179, 255, 205),
            1,
            cv2.LINE_AA,
        )

        y = 180
        cv2.putText(
            frame,
            "Top Apps During Recording",
            (42, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (235, 235, 245),
            2,
            cv2.LINE_AA,
        )
        y += 34

        if not rows:
            cv2.putText(
                frame,
                "No app activity captured.",
                (42, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 200, 210),
                1,
                cv2.LINE_AA,
            )
        else:
            max_rows = min(12, len(rows))
            for idx in range(max_rows):
                app_name, app_seconds = rows[idx]
                line = f"{idx + 1:>2}. {app_name[:52]:<52}  {format_duration(int(app_seconds))}"
                cv2.putText(
                    frame,
                    line,
                    (42, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (210, 210, 220),
                    1,
                    cv2.LINE_AA,
                )
                y += 28

        repeats = max(1, int(round(self.config.output_fps * max(0.1, seconds))))
        for _ in range(repeats):
            self._writer.write(frame)
            self._frames_written += 1


class ActiveAppTracker:
    def __init__(self, sample_interval_seconds: float = 1.0):
        self.sample_interval_seconds = max(0.2, sample_interval_seconds)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_ts: Optional[float] = None
        self._active_app = "Unknown"
        self._paused = False
        self._usage_seconds: Dict[str, int] = defaultdict(int)

    def start(self, reset: bool = True) -> None:
        if self.is_running():
            return
        if reset:
            self.reset()
        self._stop_event.clear()
        self._start_ts = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def reset(self) -> None:
        with self._lock:
            self._active_app = "Unknown"
            self._paused = False
            self._usage_seconds.clear()
            self._start_ts = None

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def elapsed_seconds(self) -> float:
        with self._lock:
            if self._start_ts is None:
                return 0.0
            return max(0.0, time.monotonic() - self._start_ts)

    def get_snapshot(self) -> Tuple[str, int, List[Tuple[str, int]]]:
        with self._lock:
            active_app = self._active_app
            rows = sorted(self._usage_seconds.items(), key=lambda item: item[1], reverse=True)
            total = sum(self._usage_seconds.values())
        return active_app, total, rows

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                paused = self._paused
            if paused:
                if self._stop_event.wait(self.sample_interval_seconds):
                    break
                continue
            app_name = self._get_active_app_name()
            with self._lock:
                self._active_app = app_name
                self._usage_seconds[app_name] += 1
            if self._stop_event.wait(self.sample_interval_seconds):
                break

    def _get_active_app_name(self) -> str:
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return "Unknown"
            pid = ctypes.c_ulong(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if not pid.value:
                return "Unknown"
            return psutil.Process(pid.value).name()
        except Exception:
            return "Unknown"


class TrayTimelapseApp:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        self.recorder: Optional[TimelapseRecorder] = None
        self.app_tracker = ActiveAppTracker(sample_interval_seconds=1.0)
        self.closing = False

        self.root = tk.Tk()
        self.root.title("Timelapse Mini Dashboard")
        self.root.geometry("700x560")
        self.root.resizable(True, True)
        self.root.configure(bg="#0F172A")
        self.root.protocol("WM_DELETE_WINDOW", self.hide_dashboard)

        self.status_var = tk.StringVar(value="IDLE")
        self.elapsed_var = tk.StringVar(value="Elapsed: 00:00:00")
        self.stats_var = tk.StringVar(value="Captured: 0 | Written: 0")
        self.scale_var = tk.StringVar(value="Auto Bar Window: 00:01:00")
        self.speed_var = tk.StringVar(value=f"{self.app_config.default_speed_factor:.2f}")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.tracker_active_app_var = tk.StringVar(value="Tracker active app: Unknown")
        self.tracker_total_var = tk.StringVar(value="Tracked app time: 00:00:00")
        self.tracker_status_var = tk.StringVar(value="Tracker: idle")

        self.start_btn: Optional[ttk.Button] = None
        self.pause_btn: Optional[ttk.Button] = None
        self.stop_btn: Optional[ttk.Button] = None
        self.apps_tree: Optional[ttk.Treeview] = None
        self._build_dashboard()
        self.hide_dashboard()

        self.tray_icon = self._create_tray_icon()
        self.tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
        self.tray_thread.start()

        self._schedule_status_refresh()

    def _build_dashboard(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Time.Horizontal.TProgressbar",
            background="#22C55E",
            troughcolor="#1E293B",
            bordercolor="#334155",
            lightcolor="#22C55E",
            darkcolor="#15803D",
            thickness=18,
        )
        style.configure(
            "Tracker.Treeview",
            background="#0B1220",
            fieldbackground="#0B1220",
            foreground="#E2E8F0",
            bordercolor="#334155",
            rowheight=26,
            relief="flat",
            font=("Segoe UI", 9),
        )
        style.map(
            "Tracker.Treeview",
            background=[("selected", "#1E3A8A")],
            foreground=[("selected", "#F8FAFC")],
        )
        style.configure(
            "Tracker.Treeview.Heading",
            background="#111827",
            foreground="#BFDBFE",
            bordercolor="#334155",
            relief="flat",
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "Tracker.Treeview.Heading",
            background=[("active", "#1F2937")],
            foreground=[("active", "#DBEAFE")],
        )
        style.configure(
            "Tracker.Vertical.TScrollbar",
            background="#1F2937",
            troughcolor="#0F172A",
            bordercolor="#334155",
            arrowcolor="#CBD5E1",
            darkcolor="#111827",
            lightcolor="#1F2937",
        )
        style.configure(
            "Session.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(14, 8),
        )

        container = tk.Frame(self.root, bg="#0F172A")
        container.pack(fill="both", expand=True)

        tk.Label(
            container,
            text="Timelapse Mini Dashboard",
            fg="#E2E8F0",
            bg="#0F172A",
            font=("Segoe UI Semibold", 13),
        ).pack(padx=14, pady=(10, 2))

        tk.Label(
            container,
            textvariable=self.status_var,
            fg="#93C5FD",
            bg="#0F172A",
            font=("Segoe UI", 11, "bold"),
        ).pack(padx=14, pady=(2, 6))

        speed_row = tk.Frame(container, bg="#0F172A")
        speed_row.pack(fill="x", padx=14, pady=(0, 8))
        tk.Label(
            speed_row,
            text="Speed Factor",
            fg="#E2E8F0",
            bg="#0F172A",
            font=("Segoe UI", 10),
        ).pack(side="left")
        tk.Entry(
            speed_row,
            textvariable=self.speed_var,
            width=10,
            justify="center",
            bg="#1E293B",
            fg="#E2E8F0",
            insertbackground="#E2E8F0",
            relief="flat",
        ).pack(side="left", padx=10)
        tk.Label(
            speed_row,
            text="(2 = faster, 0.5 = slower)",
            fg="#94A3B8",
            bg="#0F172A",
            font=("Segoe UI", 9),
        ).pack(side="left")

        ttk.Progressbar(
            container,
            orient="horizontal",
            mode="determinate",
            style="Time.Horizontal.TProgressbar",
            maximum=100.0,
            variable=self.progress_var,
            length=430,
        ).pack(padx=14, pady=(0, 8))

        controls_card = tk.Frame(container, bg="#111827", highlightbackground="#334155", highlightthickness=1)
        controls_card.pack(fill="x", padx=14, pady=(0, 8))
        tk.Label(
            controls_card,
            text="Session Controls",
            fg="#E2E8F0",
            bg="#111827",
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w", padx=10, pady=(8, 6))
        buttons = tk.Frame(controls_card, bg="#111827")
        buttons.pack(fill="x", padx=10, pady=(0, 10))
        self.start_btn = ttk.Button(
            buttons,
            text="Start Session",
            style="Session.TButton",
            command=self.start_recording,
        )
        self.start_btn.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        self.pause_btn = ttk.Button(
            buttons,
            text="Pause",
            style="Session.TButton",
            command=self.pause_resume,
            state="disabled",
        )
        self.pause_btn.grid(row=0, column=1, padx=8, sticky="ew")
        self.stop_btn = ttk.Button(
            buttons,
            text="Stop Session",
            style="Session.TButton",
            command=self.stop_recording,
            state="disabled",
        )
        self.stop_btn.grid(row=0, column=2, padx=(8, 0), sticky="ew")
        buttons.grid_columnconfigure(0, weight=1)
        buttons.grid_columnconfigure(1, weight=1)
        buttons.grid_columnconfigure(2, weight=1)

        info_wrap = tk.Frame(container, bg="#0F172A")
        info_wrap.pack(fill="x", padx=14, pady=(0, 8))
        tk.Label(info_wrap, textvariable=self.elapsed_var, fg="#E2E8F0", bg="#0F172A", font=("Consolas", 11)).pack(
            anchor="w"
        )
        tk.Label(info_wrap, textvariable=self.stats_var, fg="#A5B4FC", bg="#0F172A", font=("Segoe UI", 10)).pack(
            anchor="w"
        )
        tk.Label(info_wrap, textvariable=self.scale_var, fg="#94A3B8", bg="#0F172A", font=("Segoe UI", 9)).pack(
            anchor="w"
        )

        tk.Label(
            container,
            text="Close window to hide. Use tray icon to reopen.",
            fg="#94A3B8",
            bg="#0F172A",
            font=("Segoe UI", 9),
        ).pack(padx=14, pady=(0, 8))

        tracker_card = tk.Frame(container, bg="#111827", highlightbackground="#334155", highlightthickness=1)
        tracker_card.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        tk.Label(
            tracker_card,
            text="App Tracker (this recording)",
            fg="#E2E8F0",
            bg="#111827",
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Label(
            tracker_card,
            textvariable=self.tracker_status_var,
            fg="#86EFAC",
            bg="#111827",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=10, pady=(0, 2))
        tk.Label(
            tracker_card,
            textvariable=self.tracker_active_app_var,
            fg="#BFDBFE",
            bg="#111827",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=10, pady=(0, 2))
        tk.Label(
            tracker_card,
            textvariable=self.tracker_total_var,
            fg="#C4B5FD",
            bg="#111827",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=10, pady=(0, 6))

        columns = ("app", "used")
        tree_wrap = tk.Frame(tracker_card, bg="#111827")
        tree_wrap.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.apps_tree = ttk.Treeview(
            tree_wrap,
            columns=columns,
            show="headings",
            height=9,
            style="Tracker.Treeview",
        )
        self.apps_tree.heading("app", text="Application")
        self.apps_tree.heading("used", text="Time Used")
        self.apps_tree.column("app", width=430, anchor="w")
        self.apps_tree.column("used", width=120, anchor="e")
        tree_scroll = ttk.Scrollbar(
            tree_wrap,
            orient="vertical",
            command=self.apps_tree.yview,
            style="Tracker.Vertical.TScrollbar",
        )
        self.apps_tree.configure(yscrollcommand=tree_scroll.set)
        self.apps_tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")


    def _create_tray_image(self) -> Image.Image:
        img = Image.new("RGB", (64, 64), "#0F172A")
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((8, 8, 56, 56), radius=12, fill="#1E293B", outline="#334155", width=2)
        draw.rectangle((20, 22, 44, 28), fill="#22C55E")
        draw.rectangle((20, 34, 36, 40), fill="#93C5FD")
        return img

    def _create_tray_icon(self) -> pystray.Icon:
        menu = pystray.Menu(
            TrayItem("Show Dashboard", self._tray_show_dashboard, default=True),
            TrayItem("Start Recording", self._tray_start, enabled=lambda item: not self._is_recording()),
            TrayItem(
                lambda item: "Resume Recording" if self._is_paused() else "Pause Recording",
                self._tray_pause_resume,
                enabled=lambda item: self._is_recording(),
            ),
            TrayItem("Stop Recording", self._tray_stop, enabled=lambda item: self._is_recording()),
            pystray.Menu.SEPARATOR,
            TrayItem("Exit", self._tray_exit),
        )
        return pystray.Icon("timelapse_recorder", self._create_tray_image(), "Timelapse Recorder", menu)

    def _is_recording(self) -> bool:
        return self.recorder is not None and self.recorder.is_running()

    def _is_paused(self) -> bool:
        return self.recorder is not None and self.recorder.is_paused()

    def _update_tray_menu(self) -> None:
        if self.tray_icon:
            self.tray_icon.update_menu()

    def show_dashboard(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def hide_dashboard(self) -> None:
        self.root.withdraw()

    def _new_recorder_config(self, speed_factor: float) -> RecorderConfig:
        return RecorderConfig(
            input_fps=self.app_config.input_fps,
            speed_factor=speed_factor,
            output_fps=self.app_config.output_fps,
            monitor_index=self.app_config.monitor_index,
            temp_output_path=build_temp_output_path(self.app_config.video_format),
            default_save_name=build_default_output_name(self.app_config.video_format),
            video_format=self.app_config.video_format,
            max_width=self.app_config.max_width,
            paused_dim_alpha=self.app_config.paused_dim_alpha,
        )

    def start_recording(self) -> None:
        if self._is_recording():
            return
        try:
            speed_factor = float(self.speed_var.get().strip())
            if speed_factor <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Speed Factor", "Speed factor must be a positive number.")
            return

        recorder = TimelapseRecorder(self._new_recorder_config(speed_factor))
        recorder.start()
        self.recorder = recorder
        self.app_tracker.start(reset=True)
        self.status_var.set("RECORDING")
        self._refresh_button_states()
        self._update_tray_menu()

    def pause_resume(self) -> None:
        if not self._is_recording() or self.recorder is None:
            return
        paused = self.recorder.toggle_pause()
        self.app_tracker.set_paused(paused)
        self._refresh_button_states()
        self._update_tray_menu()

    def stop_recording(self) -> None:
        self._stop_recording_with_save_prompt()

    def _stop_recording_with_save_prompt(self) -> None:
        if self.recorder is None:
            return
        recorder = self.recorder
        active_app, total_app_seconds, app_rows = self.app_tracker.get_snapshot()
        summary_payload = {
            "active_app": active_app,
            "total_app_seconds": total_app_seconds,
            "app_rows": app_rows,
            "elapsed_seconds": recorder.elapsed_seconds(),
        }
        recorder.stop(summary_tail_payload=summary_payload)
        self.recorder = None
        self.app_tracker.stop()
        self._refresh_button_states()
        self._update_tray_menu()

        temp_path = recorder.temp_output_path()
        if not temp_path.exists():
            messagebox.showwarning("Timelapse Recorder", "Recording stopped, but no output file was created.")
            return

        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Timelapse Video",
            initialfile=recorder.config.default_save_name,
            defaultextension=f".{recorder.config.video_format}",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("All Files", "*.*")],
        )
        if selected:
            final_path = Path(selected)
            if final_path.suffix.lower() not in (".mp4", ".avi"):
                final_path = final_path.with_suffix(f".{recorder.config.video_format}")
            recorder.move_temp_to(final_path)
            messagebox.showinfo("Timelapse Recorder", f"Recording stopped.\nSaved: {final_path}")
        else:
            messagebox.showinfo(
                "Timelapse Recorder",
                "Recording stopped.\nSave cancelled.\n"
                f"Temporary file kept at: {temp_path}",
            )

    def _refresh_button_states(self) -> None:
        recording = self._is_recording()
        paused = self._is_paused()
        if recording:
            self.status_var.set("PAUSED" if paused else "RECORDING")
        else:
            self.status_var.set("IDLE")

        if self.start_btn:
            self.start_btn.config(state="disabled" if recording else "normal")
        if self.pause_btn:
            self.pause_btn.config(state="normal" if recording else "disabled")
            self.pause_btn.config(text="Resume" if paused else "Pause")
        if self.stop_btn:
            self.stop_btn.config(state="normal" if recording else "disabled")

    def _auto_scale_seconds(self, elapsed_seconds: float) -> float:
        minute_bucket = max(1, math.ceil(elapsed_seconds / 60.0))
        return float(minute_bucket * 60)

    def _schedule_status_refresh(self) -> None:
        if self.closing:
            return

        if self.recorder is not None:
            err = self.recorder.get_error()
            if err is not None:
                messagebox.showerror("Timelapse Recorder Error", err)
                self.recorder = None
                self.app_tracker.stop()
                self._refresh_button_states()
                self._update_tray_menu()
            else:
                elapsed_seconds = self.recorder.elapsed_seconds()
                elapsed = format_duration(elapsed_seconds)
                self.elapsed_var.set(f"Elapsed: {elapsed}")
                self.stats_var.set(
                    f"Captured: {self.recorder.frames_captured()} | Written: {self.recorder.frames_written()} | "
                    f"Input: {self.recorder.config.input_fps:.1f}fps | Output: {self.recorder.config.output_fps:.1f}fps | "
                    f"Speed: {self.recorder.config.speed_factor:.2f}x"
                )
                scale_seconds = self._auto_scale_seconds(max(1.0, elapsed_seconds))
                percent = min(100.0, (elapsed_seconds / scale_seconds) * 100.0)
                self.progress_var.set(percent)
                self.scale_var.set(
                    f"Auto Bar Window: {format_duration(scale_seconds)} ({percent:05.1f}% filled)"
                )
                self._refresh_button_states()
        else:
            self.elapsed_var.set("Elapsed: 00:00:00")
            self.stats_var.set("Captured: 0 | Written: 0")
            self.progress_var.set(0.0)
            self.scale_var.set("Auto Bar Window: 00:01:00 (000.0% filled)")
            self._refresh_button_states()

        self._refresh_tracker_status()

        self.root.after(500, self._schedule_status_refresh)

    def _refresh_tracker_status(self) -> None:
        if self.app_tracker.is_running():
            self.tracker_status_var.set("Tracker: running")
        else:
            self.tracker_status_var.set("Tracker: idle")

        active_app, total_seconds, rows = self.app_tracker.get_snapshot()
        self.tracker_active_app_var.set(f"Tracker active app: {active_app}")
        self.tracker_total_var.set(f"Tracked app time: {format_duration(total_seconds)}")

        if self.apps_tree is None:
            return

        self.apps_tree.delete(*self.apps_tree.get_children())
        for app_name, seconds in rows:
            self.apps_tree.insert("", "end", values=(app_name, format_duration(seconds)))

    def _tray_show_dashboard(self, _icon, _item) -> None:
        self.root.after(0, self.show_dashboard)

    def _tray_start(self, _icon, _item) -> None:
        self.root.after(0, self.start_recording)

    def _tray_pause_resume(self, _icon, _item) -> None:
        self.root.after(0, self.pause_resume)

    def _tray_stop(self, _icon, _item) -> None:
        self.root.after(0, self.stop_recording)

    def _tray_exit(self, _icon, _item) -> None:
        self.root.after(0, self.shutdown)

    def shutdown(self) -> None:
        if self.closing:
            return
        self.closing = True
        if self.recorder is not None:
            self._stop_recording_with_save_prompt()
        self.app_tracker.stop()
        self.tray_icon.stop()
        self.root.quit()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def build_default_output_name(video_format: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"timelapse_{ts}.{video_format}"


def build_temp_output_path(video_format: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("recordings") / f"_tmp_timelapse_{ts}.{video_format}"


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Timelapse tray recorder.")
    parser.add_argument("--input-fps", type=float, default=24.0, help="Capture rate in fps.")
    parser.add_argument("--output-fps", type=float, default=24.0, help="Saved video framerate.")
    parser.add_argument("--speed-factor", type=float, default=10.0, help="Default speed factor.")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (1-based).")
    parser.add_argument("--format", dest="video_format", choices=["mp4", "avi"], default="mp4")
    parser.add_argument("--max-width", type=int, default=1280, help="Downscale width. 0 disables downscale.")
    parser.add_argument("--pause-dim-alpha", type=float, default=0.58, help="Pause dim strength (0..1).")
    args = parser.parse_args()

    if args.input_fps <= 0:
        raise ValueError("--input-fps must be greater than 0.")
    if args.output_fps <= 0:
        raise ValueError("--output-fps must be greater than 0.")
    if args.speed_factor <= 0:
        raise ValueError("--speed-factor must be greater than 0.")
    if not (0.0 <= args.pause_dim_alpha <= 1.0):
        raise ValueError("--pause-dim-alpha must be between 0 and 1.")

    return AppConfig(
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        monitor_index=args.monitor,
        video_format=args.video_format,
        max_width=max(0, args.max_width),
        paused_dim_alpha=args.pause_dim_alpha,
        default_speed_factor=args.speed_factor,
    )


def main() -> None:
    app = TrayTimelapseApp(parse_args())
    app.run()


if __name__ == "__main__":
    main()
