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
        const saved: HistoryEntry[] = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
        Promise.all(
            saved.map(entry =>
                fetch(entry.imageUrl, { method: "HEAD" })
                    .then(r => (r.ok ? entry : null))
                    .catch(() => null)
            )
        ).then(results => {
            const valid = results.filter(Boolean) as HistoryEntry[];
            if (valid.length !== saved.length) {
                localStorage.setItem(HISTORY_KEY, JSON.stringify(valid));
            }
            setHistory(valid);
        });
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
                        <h2 className="font-outfit text-2xl font-bold uppercase tracking-wider" style={{ color: "var(--text-primary)" }}>Past Looks</h2>
                        <div className="w-12 h-1 bg-rose-400 mt-2" />
                    </div>
                    <button
                        onClick={clearAll}
                        className="flex items-center gap-2 text-[10px] uppercase font-bold tracking-widest hover:text-red-400 transition-colors"
                        style={{ color: "var(--text-tertiary)" }}
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
                            <div className="aspect-[3/4] rounded-2xl overflow-hidden" style={{ background: "var(--surface-secondary)" }}>
                                <img
                                    src={entry.imageUrl}
                                    alt={entry.garmentName}
                                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                                />
                            </div>
                            <button
                                onClick={e => { e.stopPropagation(); removeOne(entry.id); }}
                                className="absolute top-2 right-2 w-5 h-5 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                                style={{ background: "var(--glass)" }}
                            >
                                <X className="w-3 h-3" style={{ color: "var(--text-secondary)" }} />
                            </button>
                            <p className="text-[9px] font-semibold mt-2 uppercase tracking-wider truncate" style={{ color: "var(--text-secondary)" }}>
                                {entry.garmentName}
                            </p>
                            <p className="text-[9px] uppercase tracking-wider" style={{ color: "var(--text-tertiary)" }}>
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
                            className="absolute top-4 right-4 w-10 h-10 rounded-full flex items-center justify-center shadow-lg"
                            style={{ background: "var(--surface)" }}
                        >
                            <X className="w-5 h-5" style={{ color: "var(--text-secondary)" }} />
                        </button>
                    </div>
                </div>
            )}
        </>
    );
}
