/** Token bucket: `rate` tokens per second, up to `burst` saved. The clock is injectable for tests. */
export class Bucket {
    private tokens: number;
    private last: number;
    private readonly rate: number;
    private readonly burst: number;
    private readonly now: () => number;

    constructor(rate: number, burst: number, now: () => number = Date.now) {
        this.rate = rate;
        this.burst = burst;
        this.now = now;
        this.tokens = burst;
        this.last = now();
    }

    take(): boolean {
        const t = this.now();
        this.tokens = Math.min(this.burst, this.tokens + ((t - this.last) / 1000) * this.rate);
        this.last = t;
        if (this.tokens < 1) return false;
        this.tokens -= 1;
        return true;
    }
}
