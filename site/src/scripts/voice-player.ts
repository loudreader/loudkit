/*
  The one voice player on a page.

  Two components press play: the roster grid on Voices and Demo, and the
  single sample on the landing page. Both import this module, and a module is
  evaluated once per page, so both share the same `Audio` element and the same
  delegated click handler. That is what stops a second press from playing over
  the first, and it is why the logic lives here rather than inside either
  component's own `<script>`.

  The contract is the markup: a button with class `lk-voice-play`, a
  `data-src` to play and a `data-name` for the stop label.
*/
const player = new Audio();
let current: HTMLButtonElement | null = null;

function stop() {
  player.pause();
  if (current) {
    current.classList.remove('is-playing');
    current.setAttribute('aria-label', current.dataset.label ?? 'Play');
    current = null;
  }
}

player.addEventListener('ended', stop);
player.addEventListener('error', stop);

document.addEventListener('click', (event) => {
  const target = event.target as HTMLElement | null;
  const button = target?.closest<HTMLButtonElement>('.lk-voice-play');
  if (!button) return;

  const wasCurrent = button === current;
  stop();
  if (wasCurrent) return;

  button.dataset.label = button.getAttribute('aria-label') ?? 'Play';
  player.src = button.dataset.src ?? '';
  player.currentTime = 0;
  void player.play();
  button.classList.add('is-playing');
  button.setAttribute('aria-label', `Stop ${button.dataset.name ?? ''}`);
  current = button;
});

export {};
