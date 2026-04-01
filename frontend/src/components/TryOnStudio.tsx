"use client";

import { useState, useEffect, useRef } from "react";
import { Upload, Camera, Sparkles, Loader2, ArrowRight, ImagePlus, Download, ChevronLeft, ChevronRight, AlertTriangle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import CameraView from "./CameraView";
import { saveToHistory } from "./ResultHistory";

const API = "http://localhost:8000";

// Step labels matching backend step names
const STEP_LABELS: Record<string, string> = {
    queued: "Starting up...",
    removing_background: "Removing background",
    detecting_pose: "Detecting pose",
    parsing_body: "Parsing body shape",
    generating_tryon: "Generating try-on",
    fitting_garment: "Fitting garment",
    preserving_face: "Preserving face",
    cleaning_background: "Cleaning background",
    sharpening: "Sharpening details",
    upscaling: "Upscaling image",
    restoring_face: "Restoring face",
    saving_result: "Saving result",
    done: "Complete!",
};

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
    const [currentStep, setCurrentStep] = useState("queued");
    const [progress, setProgress] = useState(0);
    const [numSamples, setNumSamples] = useState(1);

    const [poseWarning, setPoseWarning] = useState<string | null>(null);

    const [garmentMode, setGarmentMode] = useState<GarmentMode>("gallery");
    const [customGarmentImage, setCustomGarmentImage] = useState<string | null>(null);
    const [customCategory, setCustomCategory] = useState<GarmentCategory>("one-pieces");
    const [garmentPhotoType, setGarmentPhotoType] = useState<"model" | "flat-lay">("model");

    const garmentNameRef = useRef("Custom Garment");

    useEffect(() => {
        if (!isProcessing) { setCurrentStep("queued"); setProgress(0); }
    }, [isProcessing]);

    useEffect(() => {
        if (!jobId) return;
        const poll = setInterval(async () => {
            try {
                const { data } = await axios.get(`${API}/status/${jobId}`);
                // Update progress bar and step label from backend
                if (data.step) setCurrentStep(data.step);
                if (typeof data.progress === "number") setProgress(data.progress);
                if (data.pose_warning) setPoseWarning(data.pose_warning);

                if (data.status === "done") {
                    setProgress(100);
                    setCurrentStep("done");
                    const urls: string[] = data.image_urls.map((u: string) => `${API}${u}`);
                    // Small delay so user sees 100% before results appear
                    setTimeout(() => {
                        setResultImages(urls);
                        setResultIndex(0);
                        setIsProcessing(false);
                        setJobId(null);
                    }, 600);
                    clearInterval(poll);
                    saveToHistory({ imageUrl: urls[0], garmentName: garmentNameRef.current });
                } else if (data.status === "failed") {
                    setError(data.error || "Processing failed. Please try again.");
                    setIsProcessing(false);
                    setJobId(null);
                    clearInterval(poll);
                }
            } catch { /* keep polling */ }
        }, 1500);
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
        setPoseWarning(null);
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
            if (data.pose_warning) setPoseWarning(data.pose_warning);
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

    /* Reusable pill-toggle style */
    const pillBg: React.CSSProperties = { background: "var(--surface-secondary)" };
    const pillActive: React.CSSProperties = { background: "var(--surface)" };

    return (
        <div className="backdrop-blur-md rounded-3xl p-10 mt-12 overflow-hidden relative shadow-sm"
             style={{ background: "var(--card-bg)", border: "1px solid var(--border)" }}>
            {showCamera && (
                <CameraView onCapture={img => { setUserImage(img); setShowCamera(false); }} onClose={() => setShowCamera(false)} />
            )}

            <div className="flex flex-col md:flex-row gap-12 items-start justify-center">
                {/* Person Image */}
                <div className="flex flex-col items-center gap-3 w-full max-w-[320px]">
                    <p className="text-xs font-bold uppercase tracking-widest" style={{ color: "var(--text-tertiary)" }}>Your Photo</p>
                    <div className="w-full aspect-[3/4] rounded-2xl border-2 border-dashed flex flex-col items-center justify-center relative overflow-hidden group"
                         style={{ borderColor: "var(--border)", background: "var(--surface-secondary)" }}>
                        {userImage ? (
                            <>
                                <img src={userImage} className="w-full h-full object-cover" />
                                <button onClick={() => setUserImage(null)} className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-white text-sm font-bold">
                                    Change Photo
                                </button>
                            </>
                        ) : (
                            <div className="text-center p-6">
                                <div className="mx-auto w-12 h-12 rounded-full flex items-center justify-center mb-4" style={{ background: "var(--surface)" }}>
                                    <Upload className="w-5 h-5" style={{ color: "var(--text-tertiary)" }} />
                                </div>
                                <p className="text-sm mb-6 uppercase tracking-wider font-semibold" style={{ color: "var(--text-tertiary)" }}>Upload Portrait</p>
                                <div className="flex flex-col gap-3">
                                    <label className="cursor-pointer px-6 py-3 rounded-full transition-colors text-xs font-bold uppercase tracking-widest text-center shadow-sm"
                                           style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text-secondary)" }}>
                                        Upload Photo
                                        <input type="file" className="hidden" accept="image/*" onChange={handleFileUpload} />
                                    </label>
                                    <button onClick={() => setShowCamera(true)} className="flex items-center justify-center gap-2 transition-colors text-[10px] uppercase font-bold tracking-widest"
                                            style={{ color: "var(--text-tertiary)" }}>
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
                        : <Sparkles className="w-8 h-8" style={{ color: "var(--text-tertiary)" }} />
                    }
                </div>

                {/* Garment Panel */}
                <div className="flex flex-col items-center gap-3 w-full max-w-[320px]">
                    <p className="text-xs font-bold uppercase tracking-widest" style={{ color: "var(--text-tertiary)" }}>Garment</p>

                    <div className="flex w-full rounded-full p-1 gap-1" style={pillBg}>
                        {(["gallery", "upload"] as GarmentMode[]).map(mode => (
                            <button key={mode} onClick={() => setGarmentMode(mode)}
                                className={`flex-1 py-2 rounded-full text-[11px] font-bold uppercase tracking-wider transition-all ${garmentMode === mode ? "text-rose-500 shadow-sm" : ""}`}
                                style={garmentMode === mode ? pillActive : { color: "var(--text-tertiary)" }}>
                                {mode === "gallery" ? "Collection" : "Upload Own"}
                            </button>
                        ))}
                    </div>

                    <div className="w-full aspect-[3/4] rounded-2xl border-2 relative overflow-hidden"
                         style={{ borderColor: "var(--border)", background: "var(--surface-secondary)" }}>
                        <AnimatePresence mode="wait">
                            {garmentMode === "gallery" ? (
                                <motion.div key="gallery" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="w-full h-full">
                                    {selectedGarment
                                        ? <img src={selectedGarment.image} className="w-full h-full object-cover" />
                                        : <div className="w-full h-full flex items-center justify-center text-xs font-bold uppercase tracking-widest text-center p-8" style={{ color: "var(--text-tertiary)" }}>Select a garment from the collection above</div>
                                    }
                                    <div className="absolute top-3 left-3 px-3 py-1 rounded-full backdrop-blur-md text-[10px] uppercase font-bold tracking-widest"
                                         style={{ background: "var(--glass)", border: "1px solid var(--glass-border)", color: "var(--text-secondary)" }}>Selected</div>
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
                                            <div className="mx-auto w-12 h-12 rounded-full flex items-center justify-center mb-4" style={{ background: "var(--surface)" }}>
                                                <ImagePlus className="w-5 h-5" style={{ color: "var(--text-tertiary)" }} />
                                            </div>
                                            <p className="text-sm font-semibold mb-2" style={{ color: "var(--text-tertiary)" }}>Upload Garment</p>
                                            <p className="text-[10px] uppercase tracking-wider" style={{ color: "var(--muted)" }}>PNG or JPG</p>
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
                                <button key={cat} onClick={() => setCustomCategory(cat)}
                                    className={`flex-1 py-2 rounded-full text-[10px] font-bold uppercase tracking-wider border transition-all ${customCategory === cat ? "bg-rose-500 text-white border-rose-500" : ""}`}
                                    style={customCategory !== cat ? { background: "var(--surface)", color: "var(--text-tertiary)", borderColor: "var(--border)" } : {}}>
                                    {cat === "one-pieces" ? "Dress" : cat}
                                </button>
                            ))}
                        </div>
                    )}

                    <div className="w-full flex rounded-full p-1 gap-1" style={pillBg}>
                        {(["model", "flat-lay"] as const).map(type => (
                            <button key={type} onClick={() => setGarmentPhotoType(type)}
                                className={`flex-1 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-wider transition-all ${garmentPhotoType === type ? "text-rose-500 shadow-sm" : ""}`}
                                style={garmentPhotoType === type ? pillActive : { color: "var(--text-tertiary)" }}>
                                {type === "model" ? "Worn Photo" : "Flat-Lay"}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Action area */}
            <div className="mt-12 flex flex-col items-center gap-4">
                <label className="flex items-center gap-3 cursor-pointer select-none" onClick={() => setNumSamples(p => p === 1 ? 2 : 1)}>
                    <div className={`w-10 h-5 rounded-full transition-colors relative ${numSamples === 2 ? "bg-rose-400" : ""}`}
                         style={numSamples !== 2 ? { background: "var(--border)" } : {}}>
                        <div className={`absolute top-0.5 w-4 h-4 rounded-full shadow transition-transform ${numSamples === 2 ? "translate-x-5" : "translate-x-0.5"}`}
                             style={{ background: "var(--surface)" }} />
                    </div>
                    <span className="text-[10px] uppercase font-bold tracking-widest" style={{ color: "var(--text-tertiary)" }}>
                        {numSamples === 2 ? "2 Variations (2x slower)" : "1 Result"}
                    </span>
                </label>

                <button
                    disabled={!canRun}
                    onClick={handleRunTryOn}
                    className={`px-12 py-5 rounded-full font-bold uppercase tracking-[0.2em] transition-all flex items-center gap-3 ${!canRun ? "cursor-not-allowed opacity-40" : "bg-gradient-to-r from-rose-500 to-pink-500 text-white hover:scale-105 shadow-xl shadow-rose-500/20 active:scale-95"}`}
                    style={!canRun ? { background: "var(--surface-secondary)", color: "var(--text-tertiary)" } : {}}
                >
                    {isProcessing ? "Processing..." : "Try It On"}
                    <ArrowRight className="w-5 h-5" />
                </button>

                {isProcessing && (
                    <div className="w-full max-w-md mt-4 flex flex-col items-center gap-3">
                        {/* Progress bar */}
                        <div className="w-full h-3 rounded-full overflow-hidden" style={{ background: "var(--surface-secondary)", border: "1px solid var(--border)" }}>
                            <motion.div
                                className="h-full rounded-full bg-gradient-to-r from-rose-500 to-pink-500"
                                initial={{ width: "0%" }}
                                animate={{ width: `${progress}%` }}
                                transition={{ duration: 0.5, ease: "easeOut" }}
                            />
                        </div>

                        {/* Percentage + step label */}
                        <div className="flex items-center justify-between w-full">
                            <div className="flex items-center gap-2">
                                <motion.div
                                    animate={{ rotate: 360 }}
                                    transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                                    className="w-4 h-4"
                                >
                                    <Loader2 className="w-4 h-4 text-rose-400" />
                                </motion.div>
                                <span className="text-xs font-semibold" style={{ color: "var(--text-secondary)" }}>
                                    {STEP_LABELS[currentStep] || currentStep}
                                </span>
                            </div>
                            <span className="text-sm font-bold text-rose-500">
                                {progress}%
                            </span>
                        </div>
                    </div>
                )}

                {error && <p className="text-red-400 text-xs font-medium text-center max-w-sm">{error}</p>}

                {poseWarning && !error && (
                    <div className="flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium max-w-sm text-center"
                         style={{ background: "rgba(251, 191, 36, 0.1)", border: "1px solid rgba(251, 191, 36, 0.3)", color: "#f59e0b" }}>
                        <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                        <span>{poseWarning}</span>
                    </div>
                )}
            </div>

            {/* Result overlay */}
            <AnimatePresence>
                {resultImages.length > 0 && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="absolute inset-0 z-20 backdrop-blur-2xl p-8 flex flex-col items-center justify-center"
                        style={{ background: "color-mix(in srgb, var(--background) 95%, transparent)" }}
                    >
                        <h2 className="text-3xl font-cormorant font-semibold uppercase tracking-widest mb-6 text-hero-gradient">
                            Your Look
                        </h2>

                        <div className="flex items-center gap-4 mb-3">
                            {resultImages.length > 1 && (
                                <button onClick={() => setResultIndex(i => Math.max(0, i - 1))} disabled={resultIndex === 0}
                                    className="w-10 h-10 rounded-full flex items-center justify-center disabled:opacity-30"
                                    style={{ border: "1px solid var(--border)", color: "var(--text-tertiary)" }}>
                                    <ChevronLeft className="w-5 h-5" />
                                </button>
                            )}
                            <div className="max-h-[55vh] rounded-3xl overflow-hidden shadow-2xl shadow-rose-500/10">
                                <img src={resultImages[resultIndex]} className="max-h-[55vh] max-w-full object-contain" />
                            </div>
                            {resultImages.length > 1 && (
                                <button onClick={() => setResultIndex(i => Math.min(resultImages.length - 1, i + 1))} disabled={resultIndex === resultImages.length - 1}
                                    className="w-10 h-10 rounded-full flex items-center justify-center disabled:opacity-30"
                                    style={{ border: "1px solid var(--border)", color: "var(--text-tertiary)" }}>
                                    <ChevronRight className="w-5 h-5" />
                                </button>
                            )}
                        </div>

                        {resultImages.length > 1 && (
                            <p className="text-[10px] uppercase tracking-widest mb-4" style={{ color: "var(--text-tertiary)" }}>
                                Variation {resultIndex + 1} of {resultImages.length}
                            </p>
                        )}

                        <div className="flex items-center gap-4 mt-2">
                            <button
                                onClick={() => handleDownload(resultImages[resultIndex])}
                                className="px-8 py-3 rounded-full bg-gradient-to-r from-rose-500 to-pink-500 text-white text-xs font-bold uppercase tracking-widest flex items-center gap-2 hover:scale-105 transition-transform shadow-lg shadow-rose-500/20"
                            >
                                <Download className="w-4 h-4" /> Download
                            </button>
                            <button
                                onClick={() => setResultImages([])}
                                className="px-8 py-3 rounded-full transition-colors uppercase text-xs font-bold tracking-widest"
                                style={{ border: "1px solid var(--border)", color: "var(--text-secondary)" }}
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
