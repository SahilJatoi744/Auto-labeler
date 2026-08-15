/**
 * ==========================================
 * Created by Sahil Jatoi (SJ)
 * AutoLabeler - AI Image Dataset Labeling
 * ==========================================
 */

import { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
    children?: ReactNode;
    fallback?: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null,
    };

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error('Uncaught error:', error, errorInfo);
    }

    public render() {
        if (this.state.hasError) {
            return (
                this.props.fallback || (
                    <div className="min-h-screen flex items-center justify-center bg-gray-50 p-6">
                        <div className="max-w-md w-full bg-white rounded-xl shadow-lg p-8 border border-red-100">
                            <div className="flex items-center gap-3 text-red-600 mb-4">
                                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                </svg>
                                <h2 className="text-xl font-bold">Something went wrong</h2>
                            </div>
                            <p className="text-gray-600 mb-6">
                                The application encountered an unexpected error. This has been logged for our team to investigate.
                            </p>
                            <div className="bg-red-50 rounded-lg p-4 mb-6 overflow-auto max-h-32">
                                <code className="text-xs text-red-700">
                                    {this.state.error?.toString()}
                                </code>
                            </div>
                            <button
                                onClick={() => window.location.href = '/'}
                                className="w-full py-2 bg-red-600 text-white rounded-lg font-semibold hover:bg-red-700 transition-colors"
                            >
                                Back to Dashboard
                            </button>
                        </div>
                    </div>
                )
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
