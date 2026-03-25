"use client";

import { useState, useEffect } from "react";
import GarmentCard from "@/components/GarmentCard";
import TryOnStudio from "@/components/TryOnStudio";
import ResultHistory from "@/components/ResultHistory";
import { motion } from "framer-motion";

type FilterCategory = "all" | "tops" | "bottoms" | "one-pieces";

interface Garment {
  id: string;
  name: string;
  price: string;
  image: string;
  category: string;
}

// Fallback if backend is not running
const FALLBACK_GARMENTS: Garment[] = [
  ];

export default function Home() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState<FilterCategory>("all");
  const [garments, setGarments] = useState<Garment[]>(FALLBACK_GARMENTS);

  // Fetch garments from backend; fall back to static list if unavailable
  useEffect(() => {
    fetch("http://localhost:8000/garments")
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        if (Array.isArray(data)) setGarments(data);
      })
      .catch(() => {}); // keep fallback
  }, []);

  const selectedGarment = garments.find(g => g.id === selectedId) || null;

  const filters: { label: string; value: FilterCategory }[] = [
    { label: "All", value: "all" },
    { label: "Tops", value: "tops" },
    { label: "Bottoms", value: "bottoms" },
    { label: "Dresses", value: "one-pieces" },
  ];

  const filtered = activeFilter === "all"
    ? garments
    : garments.filter((g: Garment) => g.category === activeFilter);

  return (
    <main className="pt-12 pb-20 px-6 max-w-7xl mx-auto">
      {/* Hero Section */}
      <section className="text-center mb-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <span className="text-[10px] uppercase font-black tracking-[0.5em] text-rose-400 mb-6 block">
            AI-Powered Virtual Try-On
          </span>
          <h1 className="font-cormorant text-7xl md:text-7xl font-light tracking-tight text-gray-900 mb-8">
            Virtual <span className="text-hero-gradient italic font-semibold">Try-On</span>
          </h1>
          
        </motion.div>
      </section>

      {/* Garment Grid */}
      <section id="collection" className="mb-24">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-6 mb-12">
          <div>
            <h2 className="font-outfit text-2xl font-bold text-gray-800 uppercase tracking-wider">The Collection</h2>
            <div className="w-12 h-1 bg-rose-400 mt-2" />
          </div>

          {/* Category Filter */}
          <div className="flex gap-2 bg-gray-100 p-1 rounded-full">
            {filters.map(f => (
              <button
                key={f.value}
                onClick={() => setActiveFilter(f.value)}
                className={`px-4 py-2 rounded-full text-[11px] font-bold uppercase tracking-wider transition-all ${
                  activeFilter === f.value
                    ? "bg-white text-rose-500 shadow-sm"
                    : "text-gray-400 hover:text-gray-600"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>

          <span className="text-[10px] text-gray-400 font-bold uppercase tracking-widest">
            {filtered.length} Items
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
          {filtered.map((garment: Garment) => (
            <GarmentCard
              key={garment.id}
              {...garment}
              isSelected={selectedId === garment.id}
              onSelect={setSelectedId}
            />
          ))}
        </div>
      </section>

      {/* Result History */}
      <ResultHistory />

      {/* Try-On Studio Section */}
      <section id="studio">
        <div className="flex justify-center mb-8">
          <div className="px-4 py-1 rounded-full bg-gray-100 border border-gray-200 text-[10px] font-bold uppercase tracking-[0.3em] text-gray-400">
            Studio Experience
          </div>
        </div>
        <TryOnStudio selectedGarment={selectedGarment} />
      </section>
    </main>
  );
}
