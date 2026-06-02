type FashionMarkProps = {
  className?: string;
  title?: string;
};

export default function FashionMark({ className, title = 'Fashion Imagine mark' }: FashionMarkProps) {
  return (
    <svg
      className={className}
      width="40"
      height="40"
      viewBox="0 0 40 40"
      role="img"
      aria-label={title}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <rect x="1.5" y="1.5" width="37" height="37" rx="8.5" fill="currentColor" />
      <path
        d="M20 9.2C17.7 9.2 16.25 10.7 16.25 12.55C16.25 14.15 17.3 15.15 19.05 15.78V18.4L11.6 24.78C10.95 25.34 10.82 26.27 11.29 26.98C11.62 27.47 12.16 27.75 12.75 27.75H27.25C27.84 27.75 28.38 27.47 28.71 26.98C29.18 26.27 29.05 25.34 28.4 24.78L20.95 18.4V15.05C20.95 14.58 20.63 14.19 20.18 14.08C18.98 13.79 18.25 13.31 18.25 12.55C18.25 11.78 18.89 11.2 20 11.2C21.11 11.2 21.75 11.78 21.75 12.55H23.75C23.75 10.7 22.3 9.2 20 9.2Z"
        fill="var(--color-mark-ink)"
      />
      <path
        d="M14.72 25.75H25.28L20 21.22L14.72 25.75Z"
        fill="currentColor"
        stroke="var(--color-mark-ink)"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path
        d="M30.4 9.5L25.2 30.5"
        stroke="var(--color-mark-ink)"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
      <path
        d="M29.15 14.5L31.65 15.15"
        stroke="var(--color-mark-ink)"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}
