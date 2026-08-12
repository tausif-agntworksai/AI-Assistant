/**
 * Minimal declaration for the optional `qrcode` package.
 *
 * It is an optional dependency: the two-step-verification screen draws a QR
 * code when it is installed and falls back to the setup key when it isn't.
 * Declaring the one function used here keeps the build working either way,
 * without pulling in a types package for a single call.
 */
declare module "qrcode" {
  export function toString(
    text: string,
    options?: { type?: "svg" | "terminal" | "utf8"; margin?: number; width?: number }
  ): Promise<string>;
}
