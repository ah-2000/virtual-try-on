"use client";

import { useState, useEffect } from "react";
import { Trash2, X } from "lucide-react";

const HISTORY_KEY = "couture_ai_history";
const MAX_HISTORY = 12;

export interface HistoryEntry {
    id: string;
    timestamp: number;
    imageUrl: string;
    garmentName: string;
}

/** Call this after a successful try-on to persist the result. */
export function saveToHistory(entry: Omit<HistoryEntry, "id" | "timestamp">) {
    if (typeof window === "undefined") return;
    const existing: HistoryEntry[] = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    const newEntry: HistoryEntry = { ...entry, id: crypto.randomUUID(), timestamp: Date.now() };
    localStorage.setItem(HISTORY_KEY, JSON.stringify([newEntry, ...existing].slice(0, MAX_HISTORY)));
}

export default function ResultHistory() {
    const [history, setHistory] = useState<HistoryEntry[]>([]);
    const [preview, setPreview] = useState<string | null>(null);

    useEffect(() => {
        setHistory(JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"));
    }, []);

    const clearAll = () => {
        localStorage.removeItem(HISTORY_KEY);
        setHistory([]);
    };

    const removeOne = (id: string) => {
        const updated = history.filter(e => e.id !== id);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
        setHistory(updated);
    };

    if (history.length === 0) return null;

    return (
        <>
            <section className="mb-24">
                <div className="flex justify-between items-end mb-8">
                    <div>
                        <h2 className="font-outfit text-2xl font-bold text-gray-800 uppercase tracking-wider">Past Looks</h2>
                        <div className="w-12 h-1 bg-rose-400 mt-2" />
                    </div>
                    <button
                        onClick={clearAll}
                        className="flex items-center gap-2 text-[10px] uppercase font-bold tracking-widest text-gray-400 hover:text-red-400 transition-colors"
                    >
                        <Trash2 className="w-3 h-3" />
                        Clear All
                    </button>
                </div>

                <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-4">
                    {history.map(entry => (
                        <div
                            key={entry.id}
                            className="relative group cursor-pointer"
                            onClick={() => setPreview(entry.imageUrl)}
                        >
                            <div className="aspect-[3/4] rounded-2xl overflow-hidden bg-gray-100">
                                <img
                                    src={entry.imageUrl}
                                    alt={entry.garmentName}
                                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                                />
                            </div>
                            <button
                                onClick={e => { e.stopPropagation(); removeOne(entry.id); }}
                                className="absolute top-2 right-2 w-5 h-5 rounded-full bg-white/80 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-50"
                            >
                                <X className="w-3 h-3 text-gray-500" />
                            </button>
                            <p className="text-[9px] font-semibold text-gray-500 mt-2 uppercase tracking-wider truncate">
                                {entry.garmentName}
                            </p>
                            <p className="text-[9px] text-gray-300 uppercase tracking-wider">
                                {new Date(entry.timestamp).toLocaleDateString()}
                            </p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Lightbox preview */}
            {preview && (
                <div
                    className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-8"
                    onClick={() => setPreview(null)}
                >
                    <div className="relative max-w-md w-full" onClick={e => e.stopPropagation()}>
                        <img src={preview} className="w-full rounded-3xl shadow-2xl" />
                        <button
                            onClick={() => setPreview(null)}
                            className="absolute top-4 right-4 w-10 h-10 rounded-full bg-white flex items-center justify-center shadow-lg hover:bg-gray-50"
                        >
                            <X className="w-5 h-5 text-gray-600" />
                        </button>
                    </div>
                </div>
            )}
        </>
    );
}
