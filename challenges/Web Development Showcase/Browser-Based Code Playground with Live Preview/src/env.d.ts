/// <reference types="vite/client" />

/** Source text of the packages the playground carries (see vite.config.ts), keyed by import specifier. */
declare module 'virtual:vendor' {
    const sources: Record<string, string>;
    export default sources;
}
