"use client";

import { useState } from "react";
import { Upload, Camera, Sparkles, Loader2, ArrowRight } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import CameraView from "./CameraView";

interface TryOnStudioProps {
    selectedGarment: {
        id: string;
        image: string;
        category: string;
    } | null;
}

export default function TryOnStudio({ selectedGarment }: TryOnStudioProps) {
    const [userImage, setUserImage] = useState<string | null>(null);
    const [showCamera, setShowCamera] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [resultImage, setResultImage] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => setUserImage(reader.result as string);
            reader.readAsDataURL(file);
        }
    };

    const handleCapture = (img: string) => {
        setUserImage(img);
        setShowCamera(false);
    };

    const handleRunTryOn = async () => {
        if (!userImage || !selectedGarment) return;

        setIsProcessing(true);
        setError(null);

        try {
            const formData = new FormData();

            // Convert dataURIs to blobs
            const personBlob = await (await fetch(userImage)).blob();
            const garmentBlob = await (await fetch(selectedGarment.image)).blob();

            formData.append("person_image", personBlob, "person.jpg");
            formData.append("garment_image", garmentBlob, "garment.jpg");
            formData.append("category", selectedGarment.category);

            const response = await axios.post("http://localhost:8000/tryon", formData);

            if (response.data.status === "success") {
                setResultImage(`http://localhost:8000${response.data.image_url}`);
            }
        } catch (err: any) {
            console.error(err);
            setError("Failed to run try-on. Please ensure the backend is running.");
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <div className="glass-card p-10 mt-12 overflow-hidden relative">
            {/* Camera Modal */}
            {showCamera && (
                <CameraView
                    onCapture={handleCapture}
                    onClose={() => setShowCamera(false)}
                />
            )}

            <div className="flex flex-col md:flex-row gap-12 items-center justify-center">
                {/* User Image Source */}
                <div className="w-full max-w-[320px] aspect-[3/4] rounded-2xl border-2 border-dashed border-white/10 flex flex-col items-center justify-center relative overflow-hidden group">
                    {userImage ? (
                        <>
                            <img src={userImage} className="w-full h-full object-cover" />
                            <button
                                onClick={() => setUserImage(null)}
                                className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white text-sm font-bold"
                            >
                                Change Photo
                            </button>
                        </>
                    ) : (
                        <div className="text-center p-6">
                            <div className="mx-auto w-12 h-12 rounded-full bg-white/5 flex items-center justify-center mb-4">
                                <Upload className="w-5 h-5 text-white/40" />
                            </div>
                            <p className="text-sm text-white/60 mb-6 uppercase tracking-wider font-semibold">Your Portrait</p>
                            <div className="flex flex-col gap-3">
                                <label className="cursor-pointer px-6 py-3 rounded-full bg-white/5 border border-white/10 hover:bg-white/10 transition-colors text-xs font-bold uppercase tracking-widest text-center">
                                    Upload Assets
                                    <input type="file" className="hidden" accept="image/*" onChange={handleFileUpload} />
                                </label>
                                <button
                                    onClick={() => setShowCamera(true)}
                                    className="flex items-center justify-center gap-2 text-white/40 hover:text-white transition-colors text-[10px] uppercase font-bold tracking-widest"
                                >
                                    <Camera className="w-3 h-3" />
                                    Live Shutter
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {/* Transition Icon */}
                <div className="flex flex-col items-center gap-4">
                    {isProcessing ? (
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                        >
                            <Loader2 className="w-8 h-8 text-violet-500" />
                        </motion.div>
                    ) : (
                        <Sparkles className="w-8 h-8 text-white/20" />
                    )}
                </div>

                {/* Selected Garment Preview */}
                <div className="w-full max-w-[320px] aspect-[3/4] rounded-2xl border-2 border-white/5 bg-black/20 relative overflow-hidden">
                    {selectedGarment ? (
                        <img src={selectedGarment.image} className="w-full h-full object-cover" />
                    ) : (
                        <div className="w-full h-full flex items-center justify-center text-white/20 text-xs font-bold uppercase tracking-widest text-center p-8">
                            Select a garment from the gallery above
                        </div>
                    )}
                    <div className="absolute top-4 left-4 px-3 py-1 rounded-full bg-black/60 backdrop-blur-md border border-white/10 text-[10px] uppercase font-bold text-white/60 tracking-widest">
                        Selection
                    </div>
                </div>
            </div>

            {/* Action Button */}
            <div className="mt-12 flex flex-col items-center gap-4">
                <button
                    disabled={!userImage || !selectedGarment || isProcessing}
                    onClick={handleRunTryOn}
                    className={`px-12 py-5 rounded-full font-bold uppercase tracking-[0.2em] transition-all flex items-center gap-3 ${(!userImage || !selectedGarment || isProcessing)
                        ? "bg-white/5 text-white/20 cursor-not-allowed"
                        : "bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white hover:scale-105 shadow-2xl shadow-violet-500/20 active:scale-95"
                        }`}
                >
                    {isProcessing ? "Fashioning..." : "Experience Look"}
                    <ArrowRight className="w-5 h-5" />
                </button>
                {error && <p className="text-red-400 text-xs font-medium animate-pulse">{error}</p>}
            </div>

            {/* Result Display Overlay */}
            <AnimatePresence>
                {resultImage && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        className="absolute inset-0 z-20 bg-black/80 backdrop-blur-2xl p-8 flex flex-col items-center justify-center"
                    >
                        <h2 className="text-3xl font-outfit font-bold text-gradient mb-8 uppercase tracking-widest">Your Couture Moment</h2>
                        <div className="relative w-full max-w-4xl max-h-[70vh] rounded-3xl overflow-hidden shadow-3xl shadow-violet-500/10 flex items-center justify-center">
                            <img src={resultImage} className="max-w-full max-h-full object-contain" />
                        </div>
                        <button
                            onClick={() => setResultImage(null)}
                            className="mt-8 px-10 py-4 rounded-full border border-white/20 hover:bg-white/5 transition-colors uppercase text-xs font-bold tracking-widest"
                        >
                            Back to Collection
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
