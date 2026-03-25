"use client";

import { useState, useEffect, useRef } from "react";
import { Upload, Camera, Sparkles, Loader2, ArrowRight, ImagePlus, Download, ChevronLeft, ChevronRight } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import CameraView from "./CameraView";
import { saveToHistory } from "./ResultHistory";

const API = "http://localhost:8000";

const STEPS = ["Detecting pose", "Parsing body", "Generating outfit", "Upscaling result"];
// Approximate ms when each step begins on CPU
const STEP_TIMINGS = [0, 6000, 22000, 130000];

interface TryOnStudioProps {
    selectedGarment: {
        id: string;
        name?: string;
        image: string;
        category: string;
    } | null;
}

type GarmentMode = "gallery" | "upload";
type GarmentCategory = "tops" | "bottoms" | "one-pieces";

export default function TryOnStudio({ selectedGarment }: TryOnStudioProps) {
    const [userImage, setUserImage] = useState<string | null>(null);
    const [showCamera, setShowCamera] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [jobId, setJobId] = useState<string | null>(null);
    const [resultImages, setResultImages] = useState<string[]>([]);
    const [resultIndex, setResultIndex] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [currentStep, setCurrentStep] = useState(0);
    const [numSamples, setNumSamples] = useState(1);

    const [garmentMode, setGarmentMode] = useState<GarmentMode>("gallery");
    const [customGarmentImage, setCustomGarmentImage] = useState<string | null>(null);
    const [customCategory, setCustomCategory] = useState<GarmentCategory>("one-pieces");
    const [garmentPhotoType, setGarmentPhotoType] = useState<"model" | "flat-lay">("model");

    // Capture garment name at job-start time to avoid stale closure in polling
    const garmentNameRef = useRef("Custom Garment");

    // Simulated step progress (overridden by real backend step when available)
    useEffect(() => {
        if (!isProcessing) { setCurrentStep(0); return; }
        const timers = STEP_TIMINGS.map((delay, i) =>
            setTimeout(() => setCurrentStep(prev => Math.max(prev, i)), delay)
        );
        return () => timers.forEach(clearTimeout);
    }, [isProcessing]);

    // Poll job status every 3 s
    useEffect(() => {
        if (!jobId) return;

        const poll = setInterval(async () => {
            try {
                const { data } = await axios.get(`${API}/status/${jobId}`);

                if (data.step === "upscaling") setCurrentStep(prev => Math.max(prev, 3));

                if (data.status === "done") {
                    const urls: string[] = data.image_urls.map((u: string) => `${API}${u}`);
                    setResultImages(urls);
                    setResultIndex(0);
                    setIsProcessing(false);
                    setJobId(null);
                    clearInterval(poll);
                    saveToHistory({ imageUrl: urls[0], garmentName: garmentNameRef.current });
                } else if (data.status === "failed") {
                    setError(data.error || "Processing failed. Please try again.");
                    setIsProcessing(false);
                    setJobId(null);
                    clearInterval(poll);
                }
            } catch {
                // network blip — keep polling
            }
        }, 3000);

        return () => clearInterval(poll);
    }, [jobId]);

    const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onloadend = () => setUserImage(reader.result as string);
        reader.readAsDataURL(file);
    };

    const handleGarmentUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onloadend = () => setCustomGarmentImage(reader.result as string);
        reader.readAsDataURL(file);
    };

    const activeGarment = garmentMode === "gallery"
        ? selectedGarment
        : customGarmentImage
            ? { id: "custom", name: "Custom Garment", image: customGarmentImage, category: customCategory }
            : null;

    const handleRunTryOn = async () => {
        if (!userImage || !activeGarment) return;
        setIsProcessing(true);
        setError(null);
        setResultImages([]);
        garmentNameRef.current = activeGarment.name || "Custom Garment";

        try {
            const formData = new FormData();
            const personBlob = await fetch(userImage).then(r => r.blob());
            const garmentBlob = await fetch(activeGarment.image).then(r => r.blob());
            formData.append("person_image", personBlob, "person.jpg");
            formData.append("garment_image", garmentBlob, "garment.jpg");
            formData.append("category", activeGarment.category);
            formData.append("num_samples", String(numSamples));
            formData.append("garment_photo_type", garmentPhotoType);

            const { data } = await axios.post(`${API}/tryon`, formData);
            setJobId(data.job_id);
            // isProcessing stays true — polling takes over
        } catch {
            setError("Could not reach backend. Make sure it is running on port 8000.");
            setIsProcessing(false);
        }
    };

    const handleDownload = async (url: string) => {
        const blob = await fetch(url).then(r => r.blob());
        const href = URL.createObjectURL(blob);
        Object.assign(document.createElement("a"), { href, download: "couture-ai-result.png" }).click();
        URL.revokeObjectURL(href);
    };

    const canRun = !!userImage && !!activeGarment && !isProcessing;

    return (
        <div className="bg-white/70 backdrop-blur-md border border-gray-200 rounded-3xl p-10 mt-12 overflow-hidden relative shadow-sm">
            {showCamera && (
                <CameraView onCapture={img => { setUserImage(img); setShowCamera(false); }} onClose={() => setShowCamera(false)} />
            )}

            <div className="flex flex-col md:flex-row gap-12 items-start justify-center">
                {/* Person Image */}
                <div className="flex flex-col items-center gap-3 w-full max-w-[320px]">
                    <p className="text-xs font-bold uppercase tracking-widest text-gray-400">Your Photo</p>
                    <div className="w-full aspect-[3/4] rounded-2xl border-2 border-dashed border-gray-200 flex flex-col items-center justify-center relative overflow-hidden group bg-gray-50">
                        {userImage ? (
                            <>
                                <img src={userImage} className="w-full h-full object-cover" />
                                <button onClick={() => setUserImage(null)} className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white text-sm font-bold">
                                    Change Photo
                                </button>
                            </>
                        ) : (
                            <div className="text-center p-6">
                                <div className="mx-auto w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                                    <Upload className="w-5 h-5 text-gray-400" />
                                </div>
                                <p className="text-sm text-gray-400 mb-6 uppercase tracking-wider font-semibold">Upload Portrait</p>
                                <div className="flex flex-col gap-3">
                                    <label className="cursor-pointer px-6 py-3 rounded-full bg-white border border-gray-200 hover:bg-gray-50 transition-colors text-xs font-bold uppercase tracking-widest text-gray-600 text-center shadow-sm">
                                        Upload Photo
                                        <input type="file" className="hidden" accept="image/*" onChange={handleFileUpload} />
                                    </label>
                                    <button onClick={() => setShowCamera(true)} className="flex items-center justify-center gap-2 text-gray-400 hover:text-gray-700 transition-colors text-[10px] uppercase font-bold tracking-widest">
                                        <Camera className="w-3 h-3" /> Use Camera
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Center icon */}
                <div className="flex flex-col items-center justify-center md:mt-10 gap-4 md:pt-8">
                    {isProcessing
                        ? <motion.div animate={{ rotate: 360 }} transition={{ duration: 2, repeat: Infinity, ease: "linear" }}><Loader2 className="w-8 h-8 text-rose-400" /></motion.div>
                        : <Sparkles className="w-8 h-8 text-gray-300" />
                    }
                </div>

                {/* Garment Panel */}
                <div className="flex flex-col items-center gap-3 w-full max-w-[320px]">
                    <p className="text-xs font-bold uppercase tracking-widest text-gray-400">Garment</p>

                    <div className="flex w-full rounded-full bg-gray-100 p-1 gap-1">
                        {(["gallery", "upload"] as GarmentMode[]).map(mode => (
                            <button key={mode} onClick={() => setGarmentMode(mode)} className={`flex-1 py-2 rounded-full text-[11px] font-bold uppercase tracking-wider transition-all ${garmentMode === mode ? "bg-white text-rose-500 shadow-sm" : "text-gray-400 hover:text-gray-600"}`}>
                                {mode === "gallery" ? "Collection" : "Upload Own"}
                            </button>
                        ))}
                    </div>

                    <div className="w-full aspect-[3/4] rounded-2xl border-2 border-gray-200 bg-gray-50 relative overflow-hidden">
                        <AnimatePresence mode="wait">
                            {garmentMode === "gallery" ? (
                                <motion.div key="gallery" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="w-full h-full">
                                    {selectedGarment
                                        ? <img src={selectedGarment.image} className="w-full h-full object-cover" />
                                        : <div className="w-full h-full flex items-center justify-center text-gray-300 text-xs font-bold uppercase tracking-widest text-center p-8">Select a garment from the collection above</div>
                                    }
                                    <div className="absolute top-3 left-3 px-3 py-1 rounded-full bg-white/80 backdrop-blur-md border border-gray-200 text-[10px] uppercase font-bold text-gray-500 tracking-widest">Selected</div>
                                </motion.div>
                            ) : (
                                <motion.div key="upload" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="w-full h-full flex flex-col">
                                    {customGarmentImage ? (
                                        <>
                                            <img src={customGarmentImage} className="w-full h-full object-cover" />
                                            <button onClick={() => setCustomGarmentImage(null)} className="absolute inset-0 bg-black/30 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center text-white text-sm font-bold">Change</button>
                                        </>
                                    ) : (
                                        <label className="w-full h-full flex flex-col items-center justify-center cursor-pointer p-6 text-center">
                                            <div className="mx-auto w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mb-4"><ImagePlus className="w-5 h-5 text-gray-400" /></div>
                                            <p className="text-sm text-gray-400 font-semibold mb-2">Upload Garment</p>
                                            <p className="text-[10px] text-gray-300 uppercase tracking-wider">PNG or JPG</p>
                                            <input type="file" className="hidden" accept="image/*" onChange={handleGarmentUpload} />
                                        </label>
                                    )}
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>

                    {garmentMode === "upload" && (
                        <div className="w-full flex gap-2">
                            {(["tops", "bottoms", "one-pieces"] as GarmentCategory[]).map(cat => (
                                <button key={cat} onClick={() => setCustomCategory(cat)} className={`flex-1 py-2 rounded-full text-[10px] font-bold uppercase tracking-wider border transition-all ${customCategory === cat ? "bg-rose-500 text-white border-rose-500" : "bg-white text-gray-400 border-gray-200 hover:border-rose-300"}`}>
                                    {cat === "one-pieces" ? "Dress" : cat}
                                </button>
                            ))}
                        </div>
                    )}

                    {/* Photo type toggle — shown always so gallery users can hint flat-lay */}
                    <div className="w-full flex rounded-full bg-gray-100 p-1 gap-1">
                        {(["model", "flat-lay"] as const).map(type => (
                            <button key={type} onClick={() => setGarmentPhotoType(type)} className={`flex-1 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-wider transition-all ${garmentPhotoType === type ? "bg-white text-rose-500 shadow-sm" : "text-gray-400 hover:text-gray-600"}`}>
                                {type === "model" ? "Worn Photo" : "Flat-Lay"}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Action area */}
            <div className="mt-12 flex flex-col items-center gap-4">
                {/* Variations toggle */}
                <label className="flex items-center gap-3 cursor-pointer select-none" onClick={() => setNumSamples(p => p === 1 ? 2 : 1)}>
                    <div className={`w-10 h-5 rounded-full transition-colors relative ${numSamples === 2 ? "bg-rose-400" : "bg-gray-200"}`}>
                        <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${numSamples === 2 ? "translate-x-5" : "translate-x-0.5"}`} />
                    </div>
                    <span className="text-[10px] uppercase font-bold tracking-widest text-gray-400">
                        {numSamples === 2 ? "2 Variations (2× slower)" : "1 Result"}
                    </span>
                </label>

                <button
                    disabled={!canRun}
                    onClick={handleRunTryOn}
                    className={`px-12 py-5 rounded-full font-bold uppercase tracking-[0.2em] transition-all flex items-center gap-3 ${!canRun ? "bg-gray-100 text-gray-300 cursor-not-allowed" : "bg-gradient-to-r from-rose-500 to-pink-500 text-white hover:scale-105 shadow-xl shadow-rose-200 active:scale-95"}`}
                >
                    {isProcessing ? "Processing..." : "Try It On"}
                    <ArrowRight className="w-5 h-5" />
                </button>

                {/* Processing steps */}
                {isProcessing && (
                    <div className="flex flex-wrap items-center justify-center gap-2 mt-1">
                        {STEPS.map((step, i) => (
                            <div key={step} className="flex items-center gap-1.5">
                                <div className={`w-2 h-2 rounded-full transition-all duration-500 ${i < currentStep ? "bg-rose-300" : i === currentStep ? "bg-rose-500 animate-pulse scale-125" : "bg-gray-200"}`} />
                                <span className={`text-[10px] uppercase font-bold tracking-wider transition-colors ${i === currentStep ? "text-rose-500" : i < currentStep ? "text-gray-400" : "text-gray-200"}`}>
                                    {step}
                                </span>
                                {i < STEPS.length - 1 && <span className="text-gray-200 text-xs mx-1">›</span>}
                            </div>
                        ))}
                    </div>
                )}

                {error && <p className="text-red-400 text-xs font-medium text-center max-w-sm">{error}</p>}
            </div>

            {/* Result overlay */}
            <AnimatePresence>
                {resultImages.length > 0 && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="absolute inset-0 z-20 bg-white/95 backdrop-blur-2xl p-8 flex flex-col items-center justify-center"
                    >
                        <h2 className="text-3xl font-cormorant font-semibold uppercase tracking-widest mb-6" style={{ background: "linear-gradient(135deg,#c2185b,#e91e63)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                            Your Look
                        </h2>

                        {/* Image with navigation */}
                        <div className="flex items-center gap-4 mb-3">
                            {resultImages.length > 1 && (
                                <button onClick={() => setResultIndex(i => Math.max(0, i - 1))} disabled={resultIndex === 0} className="w-10 h-10 rounded-full border border-gray-200 flex items-center justify-center text-gray-400 hover:bg-gray-50 disabled:opacity-30">
                                    <ChevronLeft className="w-5 h-5" />
                                </button>
                            )}
                            <div className="max-h-[55vh] rounded-3xl overflow-hidden shadow-2xl shadow-rose-100">
                                <img src={resultImages[resultIndex]} className="max-h-[55vh] max-w-full object-contain" />
                            </div>
                            {resultImages.length > 1 && (
                                <button onClick={() => setResultIndex(i => Math.min(resultImages.length - 1, i + 1))} disabled={resultIndex === resultImages.length - 1} className="w-10 h-10 rounded-full border border-gray-200 flex items-center justify-center text-gray-400 hover:bg-gray-50 disabled:opacity-30">
                                    <ChevronRight className="w-5 h-5" />
                                </button>
                            )}
                        </div>

                        {resultImages.length > 1 && (
                            <p className="text-[10px] uppercase tracking-widest text-gray-400 mb-4">
                                Variation {resultIndex + 1} of {resultImages.length}
                            </p>
                        )}

                        <div className="flex items-center gap-4 mt-2">
                            <button
                                onClick={() => handleDownload(resultImages[resultIndex])}
                                className="px-8 py-3 rounded-full bg-gradient-to-r from-rose-500 to-pink-500 text-white text-xs font-bold uppercase tracking-widest flex items-center gap-2 hover:scale-105 transition-transform shadow-lg shadow-rose-200"
                            >
                                <Download className="w-4 h-4" /> Download
                            </button>
                            <button
                                onClick={() => setResultImages([])}
                                className="px-8 py-3 rounded-full border border-gray-200 text-gray-600 hover:bg-gray-50 transition-colors uppercase text-xs font-bold tracking-widest"
                            >
                                Try Another
                            </button>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
