"use client";

export default function Header() {
    return (
        <header className="fixed top-0 left-0 right-0 z-50 p-6 flex justify-between items-center bg-black/50 backdrop-blur-xl border-b border-white/5">
            <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500" />
                <span className="font-outfit text-xl font-bold tracking-tighter text-gradient uppercase">
                    Virtual Try On
                </span>
            </div>
            <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-white/60">
                <a href="#" className="hover:text-white transition-colors">Spring / Summer 25</a>
                <a href="#" className="hover:text-white transition-colors">Exclusives</a>
                <a href="#" className="hover:text-white transition-colors">About</a>
            </nav>
            <div className="flex items-center gap-4">
                <button className="px-5 py-2 rounded-full border border-white/10 text-sm hover:bg-white/5 transition-colors">
                    Bespoke
                </button>
            </div>
        </header>
    );
}
