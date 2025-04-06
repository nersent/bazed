import * as React from "react";

export const Button = React.forwardRef<
  HTMLDivElement,
  React.HTMLProps<HTMLDivElement>
>(({ children, ...props }, ref) => {
  return (
    <div
      ref={ref}
      {...props}
      style={{
        width: "fit-content",
        height: "fit-content",
        padding: "0.5em 1em",
        color: "#000",
        backgroundColor: "rgba(0, 0, 0, 0.32)",
      }}
    >
      {children}
    </div>
  );
});
