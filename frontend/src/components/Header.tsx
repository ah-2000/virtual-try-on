"use client";

export default function Header() {
    return (
        <header className="fixed top-0 left-0 right-0 z-50 p-6 flex justify-between items-center bg-white/80 backdrop-blur-xl border-b border-gray-200">
            <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-pink-500 to-rose-500" />
                <span className="font-cormorant text-2xl font-semibold tracking-widest uppercase" style={{ background: "linear-gradient(135deg, #1a1a2e 0%, #c2185b 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                    Couture AI
                </span>
            </div>
            <nav className="hidden md:flex items-center gap-8 text-xs font-semibold uppercase tracking-widest text-gray-400">
                <a href="#studio" className="hover:text-gray-900 transition-colors">Try On</a>
                <a href="#collection" className="hover:text-gray-900 transition-colors">Collection</a>
            </nav>
        </header>
    );
}
