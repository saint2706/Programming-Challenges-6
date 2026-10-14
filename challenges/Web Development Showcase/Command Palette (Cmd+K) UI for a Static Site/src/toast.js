/** A short message in a polite live region, so a screen reader hears what an action did. */
export function createToast(doc = document, ms = 2500) {
    const el = doc.createElement('div');
    el.className = 'toast';
    el.setAttribute('role', 'status');
    doc.body.append(el);
    let timer;
    return (message) => {
        clearTimeout(timer);
        el.textContent = message;
        el.classList.add('show');
        timer = setTimeout(() => {
            el.classList.remove('show');
            el.textContent = '';
        }, ms);
    };
}
