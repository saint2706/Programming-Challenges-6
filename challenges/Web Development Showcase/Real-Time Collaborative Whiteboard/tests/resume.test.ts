import { expect, test } from 'vitest';
import { describe as describeBoard, Net, stroke } from './helpers.ts';

/** Reconnect scenarios that need an exact order of events, so a property test would rarely hit them. */

test('a tab that reconnects while up to date does not re-apply the last move', () => {
    const net = new Net(2);
    net.settle();
    net.docs[0]!.submit({ k: 'add', item: stroke('a') });
    net.settle();
    net.docs[0]!.submit({ k: 'mv', ids: ['a'], dx: 7, dy: 3 });
    net.settle();
    net.disconnect(1);
    net.settle(); // tab 1 comes back with last === server seq
    for (const doc of net.docs) expect(describeBoard(doc.confirmed)).toEqual(describeBoard(net.room.board));
    expect(net.room.board.items.get('a')).toMatchObject({ dx: 7, dy: 3 });
});

test('a tab that missed ops while away catches up exactly once', () => {
    const net = new Net(2);
    net.settle();
    net.docs[0]!.submit({ k: 'add', item: stroke('a') });
    net.settle();
    net.disconnect(1);
    net.docs[0]!.submit({ k: 'mv', ids: ['a'], dx: 5, dy: 0 });
    net.docs[0]!.submit({ k: 'mv', ids: ['a'], dx: 5, dy: 0 });
    net.settle();
    expect(net.docs[1]!.confirmed.items.get('a')).toMatchObject({ dx: 10, dy: 0 });
    expect(net.room.board.items.get('a')).toMatchObject({ dx: 10, dy: 0 });
});

test('ops made while offline are sent once on reconnect, even if a resend crosses the original', () => {
    const net = new Net(2);
    net.settle();
    net.docs[0]!.submit({ k: 'add', item: stroke('a') });
    net.docs[0]!.submit({ k: 'mv', ids: ['a'], dx: 4, dy: 4 });
    net.deliverUp(0); // the add reaches the room ...
    net.deliverUp(0); // ... and so does the move, but the replies are lost
    net.disconnect(0);
    net.docs[0]!.submit({ k: 'mv', ids: ['a'], dx: 1, dy: 1 }); // made offline
    net.settle();
    expect(net.room.board.items.get('a')).toMatchObject({ dx: 5, dy: 5 });
    for (const doc of net.docs) {
        expect(doc.pendingCount).toBe(0);
        expect(describeBoard(doc.confirmed)).toEqual(describeBoard(net.room.board));
    }
});

test('a tab that is behind a restarted server gets a snapshot, not a stale catch-up', () => {
    const net = new Net(1);
    net.settle();
    net.docs[0]!.submit({ k: 'add', item: stroke('a') });
    net.settle();
    net.restart();
    net.docs[0]!.submit({ k: 'add', item: stroke('b') });
    net.settle();
    expect([...net.room.board.items.keys()]).toEqual(['a', 'b']);
    expect(describeBoard(net.docs[0]!.confirmed)).toEqual(describeBoard(net.room.board));
});
