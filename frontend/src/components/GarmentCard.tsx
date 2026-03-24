"use client";

import Image from "next/image";

interface GarmentCardProps {
    id: string;
    name: string;
    price: string;
    image: string;
    category: string;
    isSelected: boolean;
    onSelect: (id: string) => void;
}

export default function GarmentCard({ id, name, price, image, isSelected, onSelect }: GarmentCardProps) {
    return (
        <div
            onClick={() => onSelect(id)}
            className={`glass-card p-4 cursor-pointer relative overflow-hidden group ${isSelected ? "ring-2 ring-violet-500 bg-white/10" : ""
                }`}
        >
            <div className="aspect-[3/4] relative rounded-xl overflow-hidden mb-4">
                <Image
                    src={image}
                    alt={name}
                    fill
                    className="object-cover transition-transform duration-700 group-hover:scale-110"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-4">
                    <span className="text-xs font-medium text-white/80 uppercase tracking-widest">Select to Try On</span>
                </div>
            </div>
            <div className="flex justify-between items-start">
                <div>
                    <h3 className="font-outfit text-sm font-semibold text-white/90">{name}</h3>
                    <p className="text-xs text-white/40 mt-1 uppercase">Spring Collection</p>
                </div>
                <span className="text-sm font-bold text-violet-400 font-outfit">{price}</span>
            </div>
            {isSelected && (
                <div className="absolute top-4 right-4 w-6 h-6 bg-violet-500 rounded-full flex items-center justify-center text-white text-[10px] font-bold">
                    ✓
                </div>
            )}
        </div>
    );
}
