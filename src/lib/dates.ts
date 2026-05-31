const formatter = new Intl.DateTimeFormat("en", {
  year: "numeric",
  month: "short",
  day: "numeric"
});

export function formatDate(date: Date): string {
  return formatter.format(date);
}
