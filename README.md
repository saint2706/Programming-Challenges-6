# Programming Challenges 6

The sixth iteration of my programming challenges repository. Same mission as before — learn new languages, explore paradigms, and build a portfolio one solved problem at a time — but a fresh challenge list, so nothing here duplicates [Pro-g-rammingChallenges4](https://github.com/saint2706/Pro-g-rammingChallenges4) or [Programming Challenges 5](https://github.com/saint2706/Programming-Challenges-5).

## About This Project

This repo is organized around five categories that reflect where I want to spend time this round: shipping practical tools, sharpening algorithms and data structures, building portfolio-grade web experiences, getting fluent in data analytics, and applying machine learning the way it's actually used in production — not just research demos.

| Category | Focus |
| --- | --- |
| **Practical Software** | Real tools you'd actually install and use day to day |
| **Algorithmic Challenges** | Classic and advanced CS problems, data structures, and complexity trade-offs |
| **Web Development Showcase** | Portfolio-worthy, interaction-heavy frontend and full-stack builds |
| **Data Analytics** | Wrangling, exploring, visualizing, and reporting on data — no predictive modeling |
| **Practical ML** | Applied, production-flavored machine learning: serving, monitoring, evaluation, MLOps |

## Progress

| Category | Completed | Total | Progress |
| --- | --- | --- | --- |
| Practical Software | 0 | 30 | ![0%](https://geps.dev/progress/0) |
| Algorithmic Challenges | 10 | 30 | ![33%](https://geps.dev/progress/33) |
| Web Development Showcase | 0 | 30 | ![0%](https://geps.dev/progress/0) |
| Data Analytics | 0 | 30 | ![0%](https://geps.dev/progress/0) |
| Practical ML | 0 | 30 | ![0%](https://geps.dev/progress/0) |
| **Total** | **10** | **150** | ![7%](https://geps.dev/progress/7) |

### Difficulty Legend

| Code | Level | Roughly means |
| --- | --- | --- |
| **B** | Beginner | A weekend project; mostly standard-library or one well-known dependency |
| **I** | Intermediate | Multiple moving parts; needs real design decisions |
| **A** | Advanced | Non-trivial architecture, performance, or correctness constraints |
| **X** | Expert | Research-adjacent, systems-level, or genuinely hard to get right |

Each category runs roughly **7 Beginner / 8 Intermediate / 8 Advanced / 7 Expert** — 30 challenges total.

## How to Contribute (A Guide for New Developers)

1. **Understand the problem** before writing code — sketch the approach, note edge cases, decide what "done" looks like.
2. **Get something working first.** Optimize and refactor after, not during.
3. **Test deliberately.** Think about what breaks your solution, not just the happy path.
4. **Re-implement in a second language** once you have a working solution — it's the fastest way to actually learn a new one.
5. **Keep going.** Contribute upstream where relevant, and let this repo grow into a real portfolio.

---

## Challenges

### 1. Practical Software

| # | Challenge | Difficulty | Notes | Status |
| --- | --- | --- | --- | --- |
| 1 | [Batch File Renamer with Regex Preview](./challenges/Practical%20Software/Batch%20File%20Renamer%20with%20Regex%20Preview/) | B | Live-preview renames before committing; support undo. | Not started |
| 2 | [Self-Hosted Paste Bin with Expiry & Burn-After-Read](./challenges/Practical%20Software/Self-Hosted%20Paste%20Bin%20with%20Expiry%20&%20Burn-After-Read/) | B | One-time-view links, TTL cleanup job, syntax highlighting. | Not started |
| 3 | [Multi-Timezone Meeting Scheduler](./challenges/Practical%20Software/Multi-Timezone%20Meeting%20Scheduler/) | B | Find overlapping working hours across N timezones; DST-aware. | Not started |
| 4 | [Offline Dictionary/Thesaurus with Fuzzy Lookup](./challenges/Practical%20Software/Offline%20Dictionary-Thesaurus%20with%20Fuzzy%20Lookup/) | B | Bundle a wordlist, use edit-distance for "did you mean". | Not started |
| 5 | [Personal Subscription Tracker with Renewal Alerts](./challenges/Practical%20Software/Personal%20Subscription%20Tracker%20with%20Renewal%20Alerts/) | B | Track recurring costs, notify before renewal, spending summary. | Not started |
| 6 | [Encrypted Diary/Journal App with Mood Tagging](./challenges/Practical%20Software/Encrypted%20Diary-Journal%20App%20with%20Mood%20Tagging/) | B | Local AES encryption, per-entry mood tag, simple trend chart. | Not started |
| 7 | [Local Network File Server with Access Logs](./challenges/Practical%20Software/Local%20Network%20File%20Server%20with%20Access%20Logs/) | B | Serve a folder over LAN, log who fetched what and when. | Not started |
| 8 | [Receipt OCR & Expense Categorizer](./challenges/Practical%20Software/Receipt%20OCR%20&%20Expense%20Categorizer/) | I | Pretrained OCR + rule/ML categorization by merchant/keywords. | Not started |
| 9 | [Self-Hosted RSS Reader with Full-Text Extraction](./challenges/Practical%20Software/Self-Hosted%20RSS%20Reader%20with%20Full-Text%20Extraction/) | I | Parse feeds, fetch full article via readability algorithm. | Not started |
| 10 | [Terminal-Based Kanban Board (ncurses)](./challenges/Practical%20Software/Terminal-Based%20Kanban%20Board%20(ncurses)/) | I | Curses UI, drag-equivalent keybindings, JSON persistence. | Not started |
| 11 | [Automatic README Generator from Repo Structure](./challenges/Practical%20Software/Automatic%20README%20Generator%20from%20Repo%20Structure/) | I | Walk the tree, infer stack from manifests, draft sections. | Not started |
| 12 | [Environment Variable / Secrets Diff Tool Across Machines](./challenges/Practical%20Software/Environment%20Variable%20-%20Secrets%20Diff%20Tool%20Across%20Machines/) | I | Compare `.env` files across hosts without leaking values. | Not started |
| 13 | [Cron Job Visual Scheduler & Failure Alerter](./challenges/Practical%20Software/Cron%20Job%20Visual%20Scheduler%20&%20Failure%20Alerter/) | I | GUI/TUI for crontab editing plus run-history and alerts. | Not started |
| 14 | [PDF Merge/Split/Watermark Toolkit](./challenges/Practical%20Software/PDF%20Merge-Split-Watermark%20Toolkit/) | I | `pypdf`/`reportlab` for page ops; batch mode via CLI. | Not started |
| 15 | [Automatic Invoice Generator & PDF Emailer for Freelancers](./challenges/Practical%20Software/Automatic%20Invoice%20Generator%20&%20PDF%20Emailer%20for%20Freelancers/) | I | Template + line items → PDF, `smtplib` to send + track. | Not started |
| 16 | [Voice Memo Transcriber & Organizer](./challenges/Practical%20Software/Voice%20Memo%20Transcriber%20&%20Organizer/) | A | Local speech-to-text (e.g. Whisper), auto-tag by topic. | Not started |
| 17 | [Meeting Notes to Action-Items Extractor](./challenges/Practical%20Software/Meeting%20Notes%20to%20Action-Items%20Extractor/) | A | NLP pass over transcripts to pull owners/dates/tasks. | Not started |
| 18 | [Git Repository Health Auditor](./challenges/Practical%20Software/Git%20Repository%20Health%20Auditor/) | A | Flag stale branches, oversized blobs, leaked secrets via scan. | Not started |
| 19 | [Terminal Dashboard for Home Network Devices](./challenges/Practical%20Software/Terminal%20Dashboard%20for%20Home%20Network%20Devices/) | A | ARP scan + uptime pings rendered as a live TUI dashboard. | Not started |
| 20 | [Automated Screenshot-to-Bug-Report Generator](./challenges/Practical%20Software/Automated%20Screenshot-to-Bug-Report%20Generator/) | A | Capture, annotate, and auto-fill a bug template from context. | Not started |
| 21 | [Multi-Account Email Digest Summarizer](./challenges/Practical%20Software/Multi-Account%20Email%20Digest%20Summarizer/) | A | IMAP across accounts, summarize unread into one daily digest. | Not started |
| 22 | [Cross-Platform Window Layout Saver/Restorer](./challenges/Practical%20Software/Cross-Platform%20Window%20Layout%20Saver-Restorer/) | A | Snapshot window positions/sizes, restore on demand. | Not started |
| 23 | [Cross-Drive Duplicate File Finder (Any File Type)](./challenges/Practical%20Software/Cross-Drive%20Duplicate%20File%20Finder%20(Any%20File%20Type)/) | A | Content-hash based, not just images; interactive cleanup UI. | Not started |
| 24 | [CLI Package Version Drift Checker Across Repos](./challenges/Practical%20Software/CLI%20Package%20Version%20Drift%20Checker%20Across%20Repos/) | X | Scan a poly-repo/monorepo for mismatched dependency versions. | Not started |
| 25 | [Voice-Controlled Home Automation Hub](./challenges/Practical%20Software/Voice-Controlled%20Home%20Automation%20Hub/) | X | Local wake-word + speech recognition driving MQTT commands. | Not started |
| 26 | [P2P LAN File Transfer Tool (No Server)](./challenges/Practical%20Software/P2P%20LAN%20File%20Transfer%20Tool%20(No%20Server)/) | X | mDNS discovery, resumable transfer, no central relay. | Not started |
| 27 | [Local LLM-Powered Shell Command Explainer & Safety Linter](./challenges/Practical%20Software/Local%20LLM-Powered%20Shell%20Command%20Explainer%20&%20Safety%20Linter/) | X | Run a small local model to explain/flag risky shell commands. | Not started |
| 28 | [Self-Updating Status Page Generator with Incident History](./challenges/Practical%20Software/Self-Updating%20Status%20Page%20Generator%20with%20Incident%20History/) | X | Poll services, publish a static status page, log incidents. | Not started |
| 29 | [Multi-Source Job Application Tracker with Auto-Parsing](./challenges/Practical%20Software/Multi-Source%20Job%20Application%20Tracker%20with%20Auto-Parsing/) | X | Parse forwarded confirmation emails into a tracked pipeline. | Not started |
| 30 | [Clipboard History Manager with Fuzzy Search](./challenges/Practical%20Software/Clipboard%20History%20Manager%20with%20Fuzzy%20Search/) | X | Persistent history across sessions, hotkey search-and-paste. | Not started |

### 2. Algorithmic Challenges

| # | Challenge | Difficulty | Notes | Status |
| --- | --- | --- | --- | --- |
| 1 | [Custom Sorting Algorithm Race Visualizer](./challenges/Algorithmic%20Challenges/Custom%20Sorting%20Algorithm%20Race%20Visualizer/) | B | Manim race across 14 algorithms (comparison, non-comparison, hybrid) on one shared array. Captions paced by subtitle-standard reading time, charging only what changed since the last one. | Implemented (Python) |
| 2 | [Balanced Bracket Validator for Multiple Bracket Types](./challenges/Algorithmic%20Challenges/Balanced%20Bracket%20Validator%20for%20Multiple%20Bracket%20Types/) | B | Configurable grammars: multi-character delimiters, opaque string/comment regions, escapes, self-pairing quotes, nesting rules, multi-error recovery, streaming. | Implemented (Python) |
| 3 | [Prime Sieve Showdown (Eratosthenes vs Atkin)](./challenges/Algorithmic%20Challenges/Prime%20Sieve%20Showdown%20(Eratosthenes%20vs%20Atkin)/) | B | Six implementations (wheel-30 segmented, mod-60 Atkin, NumPy tiers) benchmarked to 10^9 with subprocess-isolated RSS measurement, plus `primes_in_range` for windows at 10^12. | Implemented (Python) |
| 4 | [Josephus Problem Solver](./challenges/Algorithmic%20Challenges/Josephus%20Problem%20Solver/) | B | Five methods from O(n·k) simulation to the O(1) k=2 bit rotation, plus an O(k log n) solver and a Fenwick-tree elimination order. | Implemented (Python) |
| 5 | [Run-Length Encoding Compressor with Custom Alphabets](./challenges/Algorithmic%20Challenges/Run-Length%20Encoding%20Compressor%20with%20Custom%20Alphabets/) | B | Encoders closed over the alphabet: self-delimiting counts in base \|Sigma\|, four codecs, bit packing to log2(\|Sigma\|) bits, decompression-bomb guards. Beats zlib on DNA. | Implemented (Python) |
| 6 | [Anagram Grouping at Scale](./challenges/Algorithmic%20Challenges/Anagram%20Grouping%20at%20Scale/) | B | Five canonical keys benchmarked by time *and* retained bytes; a 128-bit additive multiset hash (homomorphic, O(1) insert/delete) with exact verification, out-of-core merge and hash-sharded parallelism. Unicode normalisation, grapheme clusters, and why sorting UTF-8 bytes is wrong. | Implemented (Python) |
| 7 | [Count Inversions in an Array](./challenges/Algorithmic%20Challenges/Count%20Inversions%20in%20an%20Array/) | B | Six counters including two vectorised ones: a batched-searchsorted merge sort O(n log^2 n) and a value-split radix method O(n log n) -- where the *worse* complexity wins past 300k on cache locality. Plus Kendall tau-b, the Lehmer code bijection and the Mahonian generating function. | Implemented (Python) |
| 8 | [Manacher's Algorithm for Longest Palindromic Substring](./challenges/Algorithmic%20Challenges/Manacher's%20Algorithm%20for%20Longest%20Palindromic%20Substring/) | I | d1/d2 formulation with no transformed string, so no reserved characters and any sequence works. O(1) substring palindrome queries, an eertree for distinct palindromes, O(n log n) factorisation via series links. The naive baseline is *faster* on real text and 1827x slower on a run. | Implemented (Python) |
| 9 | [Z-Algorithm String Pattern Indexer](./challenges/Algorithmic%20Challenges/Z-Algorithm%20String%20Pattern%20Indexer/) | I | Separator-free search that never copies the text and streams in O(m) memory, plus multi-pattern matching in O(L·n) where L is the *trie leaf count*, not the pattern count -- provably the minimum number of Z-scans. Borders, periods, both prefix-function conversions, and Main-Lorentz tandem repeats. | Implemented (Python) |
| 10 | [Exact String Search Benchmark](./challenges/Algorithmic%20Challenges/Exact%20String%20Search%20Benchmark/) | I | Eleven matchers, instrumented by a counting proxy rather than duplicated code, with two rankings that disagree: Boyer-Moore inspects the fewest characters and is among the slowest. Constructed worst cases for all three, including Rabin-Karp hash flooding. One transposed bit-parallel method beats `str.find`. | Implemented (Python) |
| 11 | [Minimum Spanning Tree Visualizer](./challenges/Algorithmic%20Challenges/Minimum%20Spanning%20Tree%20Visualizer/) | I | Kruskal vs Prim, animate edge selection order. | Not started |
| 12 | [N-Queens Solver with Bitmask Optimization](./challenges/Algorithmic%20Challenges/N-Queens%20Solver%20with%20Bitmask%20Optimization/) | I | Bitwise column/diagonal tracking; visualize solutions found. | Not started |
| 13 | [T9 Predictive Text (Phone Keypad) Simulator](./challenges/Algorithmic%20Challenges/T9%20Predictive%20Text%20(Phone%20Keypad)%20Simulator/) | I | Trie over digit-groups, rank by frequency. | Not started |
| 14 | [Skyline Silhouette Computation](./challenges/Algorithmic%20Challenges/Skyline%20Silhouette%20Computation/) | I | Divide-and-conquer or sweep-line over building outlines. | Not started |
| 15 | [Reservoir Sampling & Weighted Random Sampling Library](./challenges/Algorithmic%20Challenges/Reservoir%20Sampling%20&%20Weighted%20Random%20Sampling%20Library/) | I | Streaming uniform sampling plus A-Res weighted variant. | Not started |
| 16 | [Strongly Connected Components Finder](./challenges/Algorithmic%20Challenges/Strongly%20Connected%20Components%20Finder/) | A | Tarjan vs Kosaraju, benchmark on large sparse graphs. | Not started |
| 17 | [Graph Coloring Solver](./challenges/Algorithmic%20Challenges/Graph%20Coloring%20Solver/) | A | Greedy, backtracking, and Welsh-Powell, compare chromatic counts. | Not started |
| 18 | [Held-Karp Exact TSP Solver](./challenges/Algorithmic%20Challenges/Held-Karp%20Exact%20TSP%20Solver/) | A | Bitmask DP, exact solution for small n; compare vs heuristics. | Not started |
| 19 | [Multi-Sequence Alignment (Generalized LCS for K Strings)](./challenges/Algorithmic%20Challenges/Multi-Sequence%20Alignment%20(Generalized%20LCS%20for%20K%20Strings)/) | A | Extend classic 2-string LCS DP to K sequences. | Not started |
| 20 | [Course Schedule / Cycle Detection Toolkit](./challenges/Algorithmic%20Challenges/Course%20Schedule%20-%20Cycle%20Detection%20Toolkit/) | A | Kahn's vs DFS-based topological sort, detect and report cycles. | Not started |
| 21 | [Cuckoo Filter Implementation](./challenges/Algorithmic%20Challenges/Cuckoo%20Filter%20Implementation/) | A | Support deletion unlike Bloom filters; compare FP rates. | Not started |
| 22 | [Order Statistics Tree](./challenges/Algorithmic%20Challenges/Order%20Statistics%20Tree/) | A | Augmented BST for O(log n) rank/select queries. | Not started |
| 23 | [Merkle Tree Builder & Proof-of-Inclusion Verifier](./challenges/Algorithmic%20Challenges/Merkle%20Tree%20Builder%20&%20Proof-of-Inclusion%20Verifier/) | A | Build tree over a dataset, generate/verify inclusion proofs. | Not started |
| 24 | [Heavy-Light Decomposition for Tree Path Queries](./challenges/Algorithmic%20Challenges/Heavy-Light%20Decomposition%20for%20Tree%20Path%20Queries/) | X | O(log² n) path queries/updates on a static tree. | Not started |
| 25 | [Link-Cut Tree for Dynamic Tree Connectivity](./challenges/Algorithmic%20Challenges/Link-Cut%20Tree%20for%20Dynamic%20Tree%20Connectivity/) | X | Support link/cut/connectivity queries in amortized O(log n). | Not started |
| 26 | [Suffix Tree Construction (Ukkonen's Algorithm)](./challenges/Algorithmic%20Challenges/Suffix%20Tree%20Construction%20(Ukkonen's%20Algorithm)/) | X | Online O(n) construction; compare against suffix array/LCP. | Not started |
| 27 | [Minimum Cost Flow via Hungarian Algorithm](./challenges/Algorithmic%20Challenges/Minimum%20Cost%20Flow%20via%20Hungarian%20Algorithm/) | X | Solve the assignment problem optimally on a weighted bipartite graph. | Not started |
| 28 | [Persistent Segment Tree for Historical Range Queries](./challenges/Algorithmic%20Challenges/Persistent%20Segment%20Tree%20for%20Historical%20Range%20Queries/) | X | Query any past version of the structure in O(log n). | Not started |
| 29 | [External-Memory Sorting for Datasets Larger Than RAM](./challenges/Algorithmic%20Challenges/External-Memory%20Sorting%20for%20Datasets%20Larger%20Than%20RAM/) | X | K-way merge over disk-backed chunks. | Not started |
| 30 | [Approximate Nearest Neighbor Search via LSH](./challenges/Algorithmic%20Challenges/Approximate%20Nearest%20Neighbor%20Search%20via%20LSH/) | X | Locality-sensitive hashing, benchmark recall vs brute-force. | Not started |

### 3. Web Development Showcase

| # | Challenge | Difficulty | Notes | Status |
| --- | --- | --- | --- | --- |
| 1 | [Interactive Resume Timeline with Scroll Animations](./challenges/Web%20Development%20Showcase/Interactive%20Resume%20Timeline%20with%20Scroll%20Animations/) | B | Scroll-triggered reveals via Intersection Observer, no framework needed. | Not started |
| 2 | [CSS-Only 3D Card Flip Gallery](./challenges/Web%20Development%20Showcase/CSS-Only%203D%20Card%20Flip%20Gallery/) | B | Pure CSS `transform`/`perspective`, no JavaScript. | Not started |
| 3 | [Custom Cursor & Micro-Interaction Playground](./challenges/Web%20Development%20Showcase/Custom%20Cursor%20&%20Micro-Interaction%20Playground/) | B | A gallery of hover/cursor effects as reusable snippets. | Not started |
| 4 | [Animated SVG Icon Library with Hover States](./challenges/Web%20Development%20Showcase/Animated%20SVG%20Icon%20Library%20with%20Hover%20States/) | B | Hand-drawn or generated icon set with CSS/SMIL animation. | Not started |
| 5 | [Single-Page Event Invitation with RSVP](./challenges/Web%20Development%20Showcase/Single-Page%20Event%20Invitation%20with%20RSVP/) | B | Static site + serverless form handler (e.g. email/webhook). | Not started |
| 6 | [Typing-Effect Hero Banner Generator](./challenges/Web%20Development%20Showcase/Typing-Effect%20Hero%20Banner%20Generator/) | B | Configurable, embeddable widget; ship as an npm-style snippet. | Not started |
| 7 | [Responsive Email Template Builder with Live Preview](./challenges/Web%20Development%20Showcase/Responsive%20Email%20Template%20Builder%20with%20Live%20Preview/) | B | Table-based HTML email quirks, live client-simulated preview. | Not started |
| 8 | [Drag-to-Reorder Photo Album with Masonry Layout](./challenges/Web%20Development%20Showcase/Drag-to-Reorder%20Photo%20Album%20with%20Masonry%20Layout/) | I | Native drag events or a small lib; persist order. | Not started |
| 9 | [Multi-Step Wizard Form with Progress Bar](./challenges/Web%20Development%20Showcase/Multi-Step%20Wizard%20Form%20with%20Progress%20Bar/) | I | Client-side validation per step, resumable state. | Not started |
| 10 | [Client-Side Data Table with Sort/Filter/CSV Export](./challenges/Web%20Development%20Showcase/Client-Side%20Data%20Table%20with%20Sort-Filter-CSV%20Export/) | I | No backend; virtualized rows for large datasets. | Not started |
| 11 | [Interactive Resume Builder with Live PDF Export](./challenges/Web%20Development%20Showcase/Interactive%20Resume%20Builder%20with%20Live%20PDF%20Export/) | I | WYSIWYG editor rendering to a print-accurate PDF. | Not started |
| 12 | [Theme Customizer Playground](./challenges/Web%20Development%20Showcase/Theme%20Customizer%20Playground/) | I | Live CSS custom-property editor with exportable theme file. | Not started |
| 13 | [Before/After Image Comparison Slider Component](./challenges/Web%20Development%20Showcase/Before-After%20Image%20Comparison%20Slider%20Component/) | I | Draggable divider, keyboard-accessible, reusable widget. | Not started |
| 14 | [Command Palette (Cmd+K) UI for a Static Site](./challenges/Web%20Development%20Showcase/Command%20Palette%20(Cmd+K)%20UI%20for%20a%20Static%20Site/) | I | Fuzzy search over pages/actions, keyboard-first navigation. | Not started |
| 15 | [Scroll-Linked Storytelling Page](./challenges/Web%20Development%20Showcase/Scroll-Linked%20Storytelling%20Page/) | I | Scrollytelling via GSAP/Intersection Observer, pinned sections. | Not started |
| 16 | [Real-Time Collaborative Whiteboard](./challenges/Web%20Development%20Showcase/Real-Time%20Collaborative%20Whiteboard/) | A | Canvas drawing synced over WebSocket across clients. | Not started |
| 17 | [Browser-Based Code Playground with Live Preview](./challenges/Web%20Development%20Showcase/Browser-Based%20Code%20Playground%20with%20Live%20Preview/) | A | Iframe sandboxing, syntax highlighting, hot-reload output. | Not started |
| 18 | [3D Product Configurator](./challenges/Web%20Development%20Showcase/3D%20Product%20Configurator/) | A | Three.js: rotate/zoom, swap materials/colors in real time. | Not started |
| 19 | [Progressive Image Loading Gallery](./challenges/Web%20Development%20Showcase/Progressive%20Image%20Loading%20Gallery/) | A | Blur-up placeholders, lazy load, WebP fallback chain. | Not started |
| 20 | [Live Multiplayer Cursor Presence Demo](./challenges/Web%20Development%20Showcase/Live%20Multiplayer%20Cursor%20Presence%20Demo/) | A | Broadcast cursor positions/names over WebSocket to all viewers. | Not started |
| 21 | [Portfolio Site with WebGL Particle Background](./challenges/Web%20Development%20Showcase/Portfolio%20Site%20with%20WebGL%20Particle%20Background/) | A | Custom shader-driven background reacting to mouse/scroll. | Not started |
| 22 | [Accessible Design-System Documentation Site](./challenges/Web%20Development%20Showcase/Accessible%20Design-System%20Documentation%20Site/) | A | Storybook-style, built from scratch, WCAG-checked components. | Not started |
| 23 | [Real-Time Form Collaboration](./challenges/Web%20Development%20Showcase/Real-Time%20Form%20Collaboration/) | A | Google-Forms-style live co-editing with presence indicators. | Not started |
| 24 | [Offline-First PWA Recipe Box](./challenges/Web%20Development%20Showcase/Offline-First%20PWA%20Recipe%20Box/) | X | IndexedDB storage, background sync queue, conflict resolution. | Not started |
| 25 | [Custom WYSIWYG Rich Text Editor](./challenges/Web%20Development%20Showcase/Custom%20WYSIWYG%20Rich%20Text%20Editor/) | X | `contenteditable` from scratch, no third-party editor lib. | Not started |
| 26 | [Browser-Based Video Editor](./challenges/Web%20Development%20Showcase/Browser-Based%20Video%20Editor/) | X | Trim/crop/text overlay client-side via the WebCodecs API. | Not started |
| 27 | [Micro-Frontend Shell Hosting Independent Widgets](./challenges/Web%20Development%20Showcase/Micro-Frontend%20Shell%20Hosting%20Independent%20Widgets/) | X | Independently deployed/versioned widgets composed at runtime. | Not started |
| 28 | [Real-Time Multiplayer Drawing/Guessing Game](./challenges/Web%20Development%20Showcase/Real-Time%20Multiplayer%20Drawing-Guessing%20Game/) | X | Canvas + WebSocket, round timer, word selection, scoring. | Not started |
| 29 | [WebAssembly In-Browser Image Filter Studio](./challenges/Web%20Development%20Showcase/WebAssembly%20In-Browser%20Image%20Filter%20Studio/) | X | Rust/C compiled to WASM for real-time pixel filters. | Not started |
| 30 | [In-Browser Regex Tester & Visual Debugger](./challenges/Web%20Development%20Showcase/In-Browser%20Regex%20Tester%20&%20Visual%20Debugger/) | X | Live match highlighting plus a plain-English pattern explainer. | Not started |

### 4. Data Analytics

| # | Challenge | Difficulty | Notes | Status |
| --- | --- | --- | --- | --- |
| 1 | [CSV Data Profiler](./challenges/Data%20Analytics/CSV%20Data%20Profiler/) | B | Auto-detect types/nulls/distributions, emit an HTML report. | Not started |
| 2 | [Personal Spending Categorizer & Monthly Trend Report](./challenges/Data%20Analytics/Personal%20Spending%20Categorizer%20&%20Monthly%20Trend%20Report/) | B | Parse bank CSV exports, categorize, chart trends over time. | Not started |
| 3 | [Interactive Sales Dashboard from a Spreadsheet](./challenges/Data%20Analytics/Interactive%20Sales%20Dashboard%20from%20a%20Spreadsheet/) | B | Filter/drill-down client-side, no backend required. | Not started |
| 4 | [A/B Test Significance Calculator](./challenges/Data%20Analytics/A-B%20Test%20Significance%20Calculator/) | B | Chi-square and t-test with plain-English interpretation. | Not started |
| 5 | [Public API Data Aggregator & Daily Snapshot Logger](./challenges/Data%20Analytics/Public%20API%20Data%20Aggregator%20&%20Daily%20Snapshot%20Logger/) | B | Build your own longitudinal dataset by polling an API daily. | Not started |
| 6 | [Correlation Matrix Explorer for Any Uploaded Dataset](./challenges/Data%20Analytics/Correlation%20Matrix%20Explorer%20for%20Any%20Uploaded%20Dataset/) | B | Upload CSV, get an interactive heatmap with significance flags. | Not started |
| 7 | [Survey Results Visualizer](./challenges/Data%20Analytics/Survey%20Results%20Visualizer/) | B | Likert-scale charts, word clouds for open-ended answers. | Not started |
| 8 | [Cohort Retention Analysis Tool](./challenges/Data%20Analytics/Cohort%20Retention%20Analysis%20Tool/) | I | Build retention curves from raw event logs. | Not started |
| 9 | [Web Traffic Funnel Analyzer](./challenges/Data%20Analytics/Web%20Traffic%20Funnel%20Analyzer/) | I | Visualize drop-off stage by stage from clickstream data. | Not started |
| 10 | [Geo-Spatial Sales Heatmap Builder](./challenges/Data%20Analytics/Geo-Spatial%20Sales%20Heatmap%20Builder/) | I | Choropleth maps from regional aggregate data. | Not started |
| 11 | [Automated Data Quality Monitor](./challenges/Data%20Analytics/Automated%20Data%20Quality%20Monitor/) | I | Schema-drift and anomaly alerts on incoming CSV batches. | Not started |
| 12 | [Statistical Anomaly Detector for Time Series](./challenges/Data%20Analytics/Statistical%20Anomaly%20Detector%20for%20Time%20Series/) | I | Z-score/IQR/STL decomposition — statistical, not ML-based. | Not started |
| 13 | [Customer Segmentation Explorer (RFM Analysis)](./challenges/Data%20Analytics/Customer%20Segmentation%20Explorer%20(RFM%20Analysis)/) | I | Recency/Frequency/Monetary scoring with an interactive dashboard. | Not started |
| 14 | [SQL Query Performance Profiler](./challenges/Data%20Analytics/SQL%20Query%20Performance%20Profiler/) | I | Visualize and explain a query's execution plan. | Not started |
| 15 | [Multi-Source Data Reconciliation Tool](./challenges/Data%20Analytics/Multi-Source%20Data%20Reconciliation%20Tool/) | I | Diff two datasets on a key, flag and categorize mismatches. | Not started |
| 16 | [Real-Time Event Stream Dashboard](./challenges/Data%20Analytics/Real-Time%20Event%20Stream%20Dashboard/) | A | Kafka/Redis Streams feeding live-updating charts. | Not started |
| 17 | [Data Lineage Tracker](./challenges/Data%20Analytics/Data%20Lineage%20Tracker/) | A | Visualize how a column's value flows through a pipeline. | Not started |
| 18 | [Cross-Tabulation & Pivot Table Engine](./challenges/Data%20Analytics/Cross-Tabulation%20&%20Pivot%20Table%20Engine/) | A | Excel-style pivoting built from scratch, not a wrapper. | Not started |
| 19 | [Survival Analysis Toolkit](./challenges/Data%20Analytics/Survival%20Analysis%20Toolkit/) | A | Kaplan-Meier curves for churn/time-to-event analysis. | Not started |
| 20 | [Automated Report Generator](./challenges/Data%20Analytics/Automated%20Report%20Generator/) | A | Natural-language summary of a dataset's key trends. | Not started |
| 21 | [Marketing Attribution Model Comparator](./challenges/Data%20Analytics/Marketing%20Attribution%20Model%20Comparator/) | A | First-touch vs last-touch vs linear, same conversion data. | Not started |
| 22 | [Interactive Hypothesis-Testing Notebook](./challenges/Data%20Analytics/Interactive%20Hypothesis-Testing%20Notebook/) | A | Guided wizard that picks the right statistical test for you. | Not started |
| 23 | [Star-Schema Data Warehouse Builder & ETL Orchestrator](./challenges/Data%20Analytics/Star-Schema%20Data%20Warehouse%20Builder%20&%20ETL%20Orchestrator/) | A | Small-scale dimensional modeling plus a scheduled ETL job. | Not started |
| 24 | [Distributed Data Processing Pipeline](./challenges/Data%20Analytics/Distributed%20Data%20Processing%20Pipeline/) | X | Spark/Dask aggregation benchmark over large CSV/Parquet. | Not started |
| 25 | [Real-Time Anomaly Detection on Streaming Metrics](./challenges/Data%20Analytics/Real-Time%20Anomaly%20Detection%20on%20Streaming%20Metrics/) | X | Statistical process control charts on a live metric stream. | Not started |
| 26 | [Custom Columnar Storage Engine + Query Executor](./challenges/Data%20Analytics/Custom%20Columnar%20Storage%20Engine%20+%20Query%20Executor/) | X | A mini Parquet/DuckDB clone with a basic query planner. | Not started |
| 27 | [Data Catalog & Metadata Search Engine](./challenges/Data%20Analytics/Data%20Catalog%20&%20Metadata%20Search%20Engine/) | X | Auto-discover and index schemas across multiple sources. | Not started |
| 28 | [Privacy-Preserving Analytics Engine](./challenges/Data%20Analytics/Privacy-Preserving%20Analytics%20Engine/) | X | Differential privacy applied to aggregate query results. | Not started |
| 29 | [Multi-Tenant Analytics API with Result Caching](./challenges/Data%20Analytics/Multi-Tenant%20Analytics%20API%20with%20Result%20Caching/) | X | Query caching plus materialized views per tenant. | Not started |
| 30 | [Feature Store for Analytics Metrics](./challenges/Data%20Analytics/Feature%20Store%20for%20Analytics%20Metrics/) | X | Versioned, time-travel-queryable aggregate metrics. | Not started |

### 5. Practical ML

| # | Challenge | Difficulty | Notes | Status |
| --- | --- | --- | --- | --- |
| 1 | [Model Serving API with Health Checks & Logging](./challenges/Practical%20ML/Model%20Serving%20API%20with%20Health%20Checks%20&%20Logging/) | B | FastAPI + ONNX Runtime, structured request/response logging. | Not started |
| 2 | [House Price Predictor with Feature Importance Explainer](./challenges/Practical%20ML/House%20Price%20Predictor%20with%20Feature%20Importance%20Explainer/) | B | Linear/tree model plus a clear feature-contribution breakdown. | Not started |
| 3 | [Churn Prediction Dashboard for a Toy SaaS Dataset](./challenges/Practical%20ML/Churn%20Prediction%20Dashboard%20for%20a%20Toy%20SaaS%20Dataset/) | B | Train, explain, and let the user tune the decision threshold. | Not started |
| 4 | [Image Background Remover](./challenges/Practical%20ML/Image%20Background%20Remover/) | B | Pretrained segmentation model wrapped as a batch CLI tool. | Not started |
| 5 | [Duplicate Product Listing Detector](./challenges/Practical%20ML/Duplicate%20Product%20Listing%20Detector/) | B | Text + image embedding similarity for e-commerce catalogs. | Not started |
| 6 | [Resume-to-Job Matching Scorer](./challenges/Practical%20ML/Resume-to-Job%20Matching%20Scorer/) | B | TF-IDF or embedding similarity ranking, explain top matches. | Not started |
| 7 | [Email Priority Inbox Sorter](./challenges/Practical%20ML/Email%20Priority%20Inbox%20Sorter/) | B | Lightweight learned classifier, not hand-written rules. | Not started |
| 8 | [Model Drift Monitor](./challenges/Practical%20ML/Model%20Drift%20Monitor/) | I | Compare live prediction distribution against training baseline. | Not started |
| 9 | [Uplift Modeling for a Marketing Campaign](./challenges/Practical%20ML/Uplift%20Modeling%20for%20a%20Marketing%20Campaign/) | I | Two-model approach to estimate incremental campaign effect. | Not started |
| 10 | [Active-Learning-Assisted Data Labeling Tool](./challenges/Practical%20ML/Active-Learning-Assisted%20Data%20Labeling%20Tool/) | I | Suggest which unlabeled samples to label next, and why. | Not started |
| 11 | [Product Review Star-Rating Predictor with Calibration](./challenges/Practical%20ML/Product%20Review%20Star-Rating%20Predictor%20with%20Calibration/) | I | Regression/classification plus confidence calibration curves. | Not started |
| 12 | [Credit Risk Scorecard Builder](./challenges/Practical%20ML/Credit%20Risk%20Scorecard%20Builder/) | I | Binning, WOE/IV, logistic regression — the industry-standard method. | Not started |
| 13 | [Real Estate Comparable Listings Finder](./challenges/Practical%20ML/Real%20Estate%20Comparable%20Listings%20Finder/) | I | Nearest-neighbor search over engineered property features. | Not started |
| 14 | [Retail Stockout Risk Classifier](./challenges/Practical%20ML/Retail%20Stockout%20Risk%20Classifier/) | I | Predict stockout risk from sales/inventory signals (classification, not forecasting). | Not started |
| 15 | [Fraud Transaction Flagging System](./challenges/Practical%20ML/Fraud%20Transaction%20Flagging%20System/) | I | Imbalanced classification; tune the precision/recall trade-off deliberately. | Not started |
| 16 | [Feature Store Prototype](./challenges/Practical%20ML/Feature%20Store%20Prototype/) | A | Ensure online/offline feature-serving consistency. | Not started |
| 17 | [Model Versioning & Rollback System](./challenges/Practical%20ML/Model%20Versioning%20&%20Rollback%20System/) | A | Registry with canary-deployment and rollback simulation. | Not started |
| 18 | [Batch Prediction Pipeline with Data Validation Gates](./challenges/Practical%20ML/Batch%20Prediction%20Pipeline%20with%20Data%20Validation%20Gates/) | A | Great-Expectations-style checks block bad data before scoring. | Not started |
| 19 | [Learning-to-Rank System for a Content Feed](./challenges/Practical%20ML/Learning-to-Rank%20System%20for%20a%20Content%20Feed/) | A | LambdaMART or similar, optimize a ranking metric directly. | Not started |
| 20 | [Cold-Start Recommendation Strategy Comparator](./challenges/Practical%20ML/Cold-Start%20Recommendation%20Strategy%20Comparator/) | A | Content-based vs popularity vs hybrid, measured on new users/items. | Not started |
| 21 | [Automated Model Retraining Trigger System](./challenges/Practical%20ML/Automated%20Model%20Retraining%20Trigger%20System/) | A | Monitor metric decay, automatically kick off a retrain job. | Not started |
| 22 | [Synthetic Tabular Data Generator for Privacy-Safe Testing](./challenges/Practical%20ML/Synthetic%20Tabular%20Data%20Generator%20for%20Privacy-Safe%20Testing/) | A | SMOTE-lite/CTGAN-style generation preserving statistical properties. | Not started |
| 23 | [Multi-Model Ensemble Voting Service](./challenges/Practical%20ML/Multi-Model%20Ensemble%20Voting%20Service/) | A | Confidence-weighted aggregation across heterogeneous models. | Not started |
| 24 | [End-to-End ML Pipeline with CI/CD](./challenges/Practical%20ML/End-to-End%20ML%20Pipeline%20with%20CI-CD/) | X | Train → validate → deploy → monitor, fully automated. | Not started |
| 25 | [Real-Time Fraud Scoring at the Edge](./challenges/Practical%20ML/Real-Time%20Fraud%20Scoring%20at%20the%20Edge/) | X | Sub-10ms inference via model quantization/optimization. | Not started |
| 26 | [Federated Learning Simulator](./challenges/Practical%20ML/Federated%20Learning%20Simulator/) | X | Train across simulated distributed clients, no raw data sharing. | Not started |
| 27 | [Model Card & Bias Audit Generator](./challenges/Practical%20ML/Model%20Card%20&%20Bias%20Audit%20Generator/) | X | Fairness metrics broken out across demographic slices. | Not started |
| 28 | [LLM Fine-Tuning Pipeline for a Domain-Specific Task](./challenges/Practical%20ML/LLM%20Fine-Tuning%20Pipeline%20for%20a%20Domain-Specific%20Task/) | X | LoRA/QLoRA on an open model for a narrow, real task. | Not started |
| 29 | [Hybrid Retrieval System for RAG](./challenges/Practical%20ML/Hybrid%20Retrieval%20System%20for%20RAG/) | X | BM25 + dense embeddings + a re-ranker model, production-style. | Not started |
| 30 | [Cost-Aware Model Selection Router](./challenges/Practical%20ML/Cost-Aware%20Model%20Selection%20Router/) | X | Route requests to a cheap or expensive model by difficulty. | Not started |

---

## Further Learning

Previous iterations of this project: [Pro-g-rammingChallenges4](https://github.com/saint2706/Pro-g-rammingChallenges4) · [Programming Challenges 5](https://github.com/saint2706/Programming-Challenges-5)

### Recommended Reading

- Cormen et al.: *Introduction to Algorithms*
- Skiena: *The Algorithm Design Manual*
- Kleppmann: *Designing Data-Intensive Applications*
- Huyen: *Designing Machine Learning Systems*
- Wickham & Grolemund: *R for Data Science* (the methodology generalizes beyond R)

### More Challenges

- [Rosetta Code](https://www.rosettacode.org)
- [Project Euler](https://www.projecteuler.net)
- [Advent of Code](https://adventofcode.com)
- [Kaggle](https://www.kaggle.com)
