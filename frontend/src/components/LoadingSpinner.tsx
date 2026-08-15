/**
 * ==========================================
 * Created by Sahil Jatoi (SJ)
 * AutoLabeler - AI Image Dataset Labeling
 * ==========================================
 */

import React from 'react';

interface LoadingSpinnerProps {
    size?: 'sm' | 'md' | 'lg';
    color?: string;
    label?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
    size = 'md',
    color = 'blue',
    label
}) => {
    const sizeClasses = {
        sm: 'h-4 w-4 border-2',
        md: 'h-8 w-8 border-3',
        lg: 'h-12 w-12 border-4',
    };

    const colorClasses = {
        blue: 'border-blue-600',
        white: 'border-white',
        gray: 'border-gray-300',
    };

    return (
        <div className="flex flex-col items-center justify-center gap-3">
            <div
                className={`
          animate-spin rounded-full border-t-transparent 
          ${sizeClasses[size]} 
          ${colorClasses[color as keyof typeof colorClasses] || colorClasses.blue}
        `}
            />
            {label && <p className="text-sm font-medium text-gray-500">{label}</p>}
        </div>
    );
};

export default LoadingSpinner;
