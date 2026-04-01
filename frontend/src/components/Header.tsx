"use client";

import { useTheme } from "./ThemeProvider";
import { Moon, Sun } from "lucide-react";

export default function Header() {
    const { theme, toggle } = useTheme();

    return (
        <header className="fixed top-0 left-0 right-0 z-50 p-6 flex justify-between items-center backdrop-blur-xl border-b"
            style={{ background: "var(--glass)", borderColor: "var(--border)" }}
        >
            <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-pink-500 to-rose-500" />
                <span className="font-cormorant text-2xl font-semibold tracking-widest uppercase text-hero-gradient">
                    Couture AI
                </span>
            </div>

            <div className="flex items-center gap-6">
                <nav className="hidden md:flex items-center gap-8 text-xs font-semibold uppercase tracking-widest"
                    style={{ color: "var(--text-tertiary)" }}
                >
                    <a href="#studio" className="hover:opacity-100 opacity-60 transition-opacity">Try On</a>
                    <a href="#collection" className="hover:opacity-100 opacity-60 transition-opacity">Collection</a>
                </nav>

                <button
                    onClick={toggle}
                    className="relative w-14 h-7 rounded-full p-0.5 transition-all duration-500 overflow-hidden"
                    style={{ background: theme === "dark" ? "linear-gradient(135deg, #1e1b4b, #4c1d95)" : "linear-gradient(135deg, #fef3c7, #fde68a)" }}
                    aria-label="Toggle dark mode"
                >
                    <div
                        className="w-6 h-6 rounded-full flex items-center justify-center transition-all duration-500 shadow-lg"
                        style={{
                            transform: theme === "dark" ? "translateX(28px)" : "translateX(0)",
                            background: theme === "dark" ? "#0f0f1a" : "#ffffff",
                        }}
                    >
                        {theme === "dark"
                            ? <Moon className="w-3.5 h-3.5 text-violet-400" />
                            : <Sun className="w-3.5 h-3.5 text-amber-500" />
                        }
                    </div>
                </button>
            </div>
        </header>
    );
}
