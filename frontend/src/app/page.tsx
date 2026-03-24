"use client";

import { useState } from "react";
import Header from "@/components/Header";
import GarmentCard from "@/components/GarmentCard";
import TryOnStudio from "@/components/TryOnStudio";
import { motion } from "framer-motion";

const GARMENTS = [
  { id: "3", name: "Pastel Silk Summer", price: "$420", image: "/garments/summer.png", category: "one-pieces" },
  { id: "4", name: "Lace Cocktail Noir", price: "$680", image: "/garments/cocktail.png", category: "one-pieces" },
  { id: "5", name: "Ivory Satin Muse", price: "$550", image: "/garments/new_dress_1.png", category: "one-pieces" },
  { id: "6", name: "Midnight Aurora", price: "$980", image: "/garments/new_dress_2.png", category: "one-pieces" },
  { id: "7", name: "Boho Ethereal Midi", price: "$320", image: "/garments/new_dress_3.jpg", category: "one-pieces" },
  { id: "8", name: "Modern Minimalist", price: "$280", image: "/garments/new_dress_4.png", category: "tops" },
  { id: "9", name: "Ethereal Silk Wrap", price: "$490", image: "/garments/garment.webp", category: "tops" },
  { id: "10", name: "New Garment 1", price: "$500", image: "/garments/Screenshot 2026-03-16 at 11.16.37 AM.png", category: "one-pieces" },
  { id: "11", name: "New Garment 2", price: "$600", image: "/garments/Screenshot 2026-03-16 at 11.16.57 AM.png", category: "one-pieces" },
  { id: "12", name: "New Garment 3", price: "$700", image: "/garments/Screenshot 2026-03-16 at 11.17.20 AM.png", category: "one-pieces" },
  { id: "13", name: "New Garment 4", price: "$450", image: "/garments/Screenshot 2026-03-16 at 1.16.22 PM.png", category: "one-pieces" },
  { id: "14", name: "New Garment 5", price: "$550", image: "/garments/Screenshot 2026-03-16 at 1.17.30 PM.png", category: "one-pieces" },
  { id: "15", name: "New Garment 6", price: "$490", image: "/garments/Screenshot 2026-03-16 at 3.44.31 PM.png", category: "one-pieces" },
  { id: "16", name: "New Garment 7", price: "$520", image: "/garments/Screenshot 2026-03-16 at 3.46.09 PM.png", category: "one-pieces" },
];

export default function Home() {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const selectedGarment = GARMENTS.find(g => g.id === selectedId) || null;

  return (
    <main className="pt-32 pb-20 px-6 max-w-7xl mx-auto">
      <Header />

      {/* Hero Section */}
      <section className="text-center mb-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <span className="text-[10px] uppercase font-black tracking-[0.5em] text-violet-400 mb-6 block">
            The Future of Haute Couture
          </span>
          <h1 className="font-outfit text-6xl md:text-8xl font-black tracking-tighter text-white mb-8">
            Virtual <span className="text-gradient italic">Studio</span>
          </h1>
          <p className="max-w-2xl mx-auto text-white/40 text-sm md:text-lg leading-relaxed font-medium">
            Step into the future of fashion. Select an exquisite piece from our Spring Collection,
            capture your portrait, and witness the magic of AI couture transformation.
          </p>
        </motion.div>
      </section>

      {/* Garment Grid */}
      <section className="mb-24">
        <div className="flex justify-between items-end mb-12">
          <div>
            <h2 className="font-outfit text-2xl font-bold text-white uppercase tracking-wider">The Collection</h2>
            <div className="w-12 h-1 bg-violet-500 mt-2" />
          </div>
          <span className="text-[10px] text-white/40 font-bold uppercase tracking-widest">14 Items Displayed</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {GARMENTS.map((garment) => (
            <GarmentCard
              key={garment.id}
              {...garment}
              isSelected={selectedId === garment.id}
              onSelect={setSelectedId}
            />
          ))}
        </div>
      </section>

      {/* Try-On Studio Section */}
      <section id="studio">
        <div className="flex justify-center mb-8">
          <div className="px-4 py-1 rounded-full bg-white/5 border border-white/10 text-[10px] font-bold uppercase tracking-[0.3em] text-white/40">
            Studio Experience
          </div>
        </div>
        <TryOnStudio selectedGarment={selectedGarment} />
      </section>

      {/* Footer Branding */}
      <footer className="mt-40 text-center border-t border-white/5 pt-20">
        <p className="text-[10px] uppercase font-black tracking-[1em] text-white/10">
          Zamna Boutique &copy; 2025
        </p>
      </footer>
    </main>
  );
}
