"use client";

import { useEffect, useRef, useState } from "react";
import { Camera, RefreshCw, Check, X, Timer } from "lucide-react";

interface CameraViewProps {
    onCapture: (image: string) => void;
    onClose: () => void;
}

type TimerOption = 0 | 5 | 10;

export default function CameraView({ onCapture, onClose }: CameraViewProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [capturedImage, setCapturedImage] = useState<string | null>(null);
    const [timerDuration, setTimerDuration] = useState<TimerOption>(0);
    const [countdown, setCountdown] = useState<number | null>(null);
    const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);

    useEffect(() => {
        async function startCamera() {
            try {
                const s = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "user", width: 1280, height: 720 }
                });
                setStream(s);
                if (videoRef.current) videoRef.current.srcObject = s;
            } catch (err) {
                console.error("Camera access denied:", err);
                alert("Please enable camera access to take a photo.");
                onClose();
            }
        }
        startCamera();
        return () => {
            stream?.getTracks().forEach(track => track.stop());
            if (countdownRef.current) clearInterval(countdownRef.current);
        };
    }, []);

    const takePhoto = () => {
        if (!videoRef.current) return;
        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
            // Flip horizontally so the captured image matches the mirrored preview
            ctx.translate(canvas.width, 0);
            ctx.scale(-1, 1);
            ctx.drawImage(videoRef.current, 0, 0);
            const data = canvas.toDataURL("image/jpeg");
            setCapturedImage(data);
        }
    };

    const startTimerWithDuration = (seconds: number) => {
        if (seconds === 0) {
            takePhoto();
            return;
        }

        setCountdown(seconds);
        let remaining = seconds;

        countdownRef.current = setInterval(() => {
            remaining -= 1;
            if (remaining <= 0) {
                if (countdownRef.current) clearInterval(countdownRef.current);
                countdownRef.current = null;
                setCountdown(null);
                takePhoto();
            } else {
                setCountdown(remaining);
            }
        }, 1000);
    };

    const cancelTimer = () => {
        if (countdownRef.current) clearInterval(countdownRef.current);
        countdownRef.current = null;
        setCountdown(null);
    };

    const confirmPhoto = () => {
        if (capturedImage) onCapture(capturedImage);
    };

    const timerOptions: TimerOption[] = [0, 5, 10];

    return (
        <div className="fixed inset-0 z-[100] bg-black/90 flex flex-col items-center justify-center p-4">
            <div className="relative w-full max-w-2xl aspect-video rounded-3xl overflow-hidden glass-card">
                {!capturedImage ? (
                    <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover scale-x-[-1]" />
                ) : (
                    <img src={capturedImage} className="w-full h-full object-cover" alt="Captured" />
                )}

                {/* Overlays */}
                <div className="absolute inset-0 pointer-events-none border-[40px] border-black/20" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-64 border-2 border-white/20 border-dashed rounded-full" />

                {/* Countdown overlay */}
                {countdown !== null && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/40 pointer-events-none">
                        <span className="text-8xl font-bold text-white drop-shadow-lg animate-pulse">
                            {countdown}
                        </span>
                    </div>
                )}

                <button onClick={onClose} className="absolute top-6 right-6 p-2 rounded-full bg-black/40 hover:bg-black/60 text-white transition-colors pointer-events-auto">
                    <X className="w-6 h-6" />
                </button>
            </div>

            {/* Controls */}
            <div className="mt-8 flex flex-col items-center gap-5">
                {!capturedImage && countdown === null && (
                    <>
                        {/* Timer options: Off = manual, 5s/10s = auto capture */}
                        <div className="flex items-center gap-2">
                            <Timer className="w-4 h-4 text-white/50" />
                            {timerOptions.map(t => (
                                <button
                                    key={t}
                                    onClick={() => {
                                        setTimerDuration(t);
                                        if (t > 0) startTimerWithDuration(t);
                                    }}
                                    className={`px-4 py-1.5 rounded-full text-[11px] font-bold uppercase tracking-wider transition-all ${
                                        timerDuration === t
                                            ? "bg-white text-black"
                                            : "bg-white/10 text-white/60 hover:bg-white/20"
                                    }`}
                                >
                                    {t === 0 ? "Off" : `${t}s`}
                                </button>
                            ))}
                        </div>

                        {/* Manual capture button — only shown when timer is Off */}
                        <button
                            onClick={takePhoto}
                            className="w-20 h-20 rounded-full bg-white flex items-center justify-center p-1 border-4 border-white/20 hover:scale-105 transition-transform"
                        >
                            <div className="w-full h-full rounded-full border-2 border-black/10 flex items-center justify-center">
                                <Camera className="w-8 h-8 text-black" />
                            </div>
                        </button>
                    </>
                )}

                {/* Countdown active — show cancel */}
                {countdown !== null && (
                    <button
                        onClick={cancelTimer}
                        className="px-6 py-3 rounded-full bg-white/10 text-white hover:bg-white/20 transition-colors text-xs font-bold uppercase tracking-widest"
                    >
                        Cancel
                    </button>
                )}

                {/* After capture — retake / use */}
                {capturedImage && (
                    <div className="flex items-center gap-6">
                        <button
                            onClick={() => setCapturedImage(null)}
                            className="p-4 rounded-full bg-white/10 text-white hover:bg-white/20 transition-colors flex items-center gap-2"
                        >
                            <RefreshCw className="w-6 h-6" />
                            <span>Retake</span>
                        </button>
                        <button
                            onClick={confirmPhoto}
                            className="px-8 py-4 rounded-full bg-violet-600 text-white hover:bg-violet-700 transition-all font-bold flex items-center gap-2 shadow-lg shadow-violet-500/20"
                        >
                            <Check className="w-6 h-6" />
                            <span>Use Photo</span>
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}
