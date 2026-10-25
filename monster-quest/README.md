# Wildbound: Veyra

`index.html` is a complete, self-contained HTML5 Canvas monster-collecting RPG. Open it directly in a browser; it has no build step, libraries, external images, or network requirements.

## Controls

- **Arrow keys / WASD:** move, navigate menus
- **Z / Enter:** confirm, interact, advance dialogue
- **X / Escape:** cancel, close menus
- **P:** pause
- Touch buttons are included below the canvas for mobile play.

## Included game flow

1. Title screen, new game, continue, and localStorage save.
2. Professor introduction and three original starter creatures.
3. Aster Town, Route 1, Mossveil Forest, Shiverstone Cave, Lumina City, Healing Center, Clover Shop, three gyms, and Veyra Summit.
4. Smooth tile movement, collision, camera following, doors/warps, tall-grass encounters, line-of-sight trainer challenges, and dialogue typewriter text.
5. 19 original creatures, six types, evolutions, four-move parties, PC box overflow, XP, level growth, and evolution animation.
6. Turn-based battles with Fight, Bag, Switch, and Run; PP, accuracy, status effects, critical hits, type effectiveness, enemy move selection, catching, trainer rewards, and animated HP bars.
7. Potions, revive, status cures, two ball tiers, buying/selling, healing, three badge gates, rival, gym leaders, champion, and ending screen.
8. Canvas pixel-art rendering and oscillator sound effects/music with a mute option.

## Add content

- Add a move to `MOVES` with `power`, `acc`, `pp`, and optional `status`, `chance`, or `heal` fields.
- Add a creature to `CREATURES` with `types`, base stats, a four-item `moves` array, and optional `evo`/`to` fields.
- Add encounters or NPCs to a map in `makeMaps()`.
- Add a map by creating a new grid, assigning warps/NPCs, and adding it to the `MAPS` object.

The code is organized with comments for **data**, **maps**, **engine state**, **rendering**, **world**, **battle**, **input/menus**, and the **game loop**. The implementation intentionally keeps all content in one file so it can be copied, opened, and edited without a toolchain.
