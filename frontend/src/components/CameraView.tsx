"use client";

import { useEffect, useRef, useState } from "react";
import { Camera, RefreshCw, Check, X } from "lucide-react";

interface CameraViewProps {
    onCapture: (image: string) => void;
    onClose: () => void;
}

export default function CameraView({ onCapture, onClose }: CameraViewProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [stream, setStream] = useState<MediaStream | null>(null);
    const [capturedImage, setCapturedImage] = useState<string | null>(null);

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
        return () => stream?.getTracks().forEach(track => track.stop());
    }, []);

    const takePhoto = () => {
        if (!videoRef.current) return;
        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
            ctx.drawImage(videoRef.current, 0, 0);
            const data = canvas.toDataURL("image/jpeg");
            setCapturedImage(data);
        }
    };

    const confirmPhoto = () => {
        if (capturedImage) onCapture(capturedImage);
    };

    return (
        <div className="fixed inset-0 z-[100] bg-black/90 flex flex-col items-center justify-center p-4">
            <div className="relative w-full max-w-2xl aspect-video rounded-3xl overflow-hidden glass-card">
                {!capturedImage ? (
                    <video ref={videoRef} autoPlay playsInline className="w-full h-full object-cover scale-x-[-1]" />
                ) : (
                    <img src={capturedImage} className="w-full h-full object-cover scale-x-[-1]" />
                )}

                {/* Overlays */}
                <div className="absolute inset-0 pointer-events-none border-[40px] border-black/20" />
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-64 border-2 border-white/20 border-dashed rounded-full" />

                <button onClick={onClose} className="absolute top-6 right-6 p-2 rounded-full bg-black/40 hover:bg-black/60 text-white transition-colors pointer-events-auto">
                    <X className="w-6 h-6" />
                </button>
            </div>

            <div className="mt-12 flex items-center gap-6">
                {!capturedImage ? (
                    <button
                        onClick={takePhoto}
                        className="w-20 h-20 rounded-full bg-white flex items-center justify-center p-1 border-4 border-white/20 hover:scale-105 transition-transform"
                    >
                        <div className="w-full h-full rounded-full border-2 border-black/10 flex items-center justify-center">
                            <Camera className="w-8 h-8 text-black" />
                        </div>
                    </button>
                ) : (
                    <>
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
                    </>
                )}
            </div>
        </div>
    );
}
