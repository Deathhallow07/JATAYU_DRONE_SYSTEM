import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";

// One .env for the whole app — the same Python/.env the relay and the
// producers read. Vite would otherwise look for it in THIS directory, so a key
// written alongside every other path would be silently ignored and the console
// would keep using a built-in default while the file said otherwise.
const ENV_DIR = "../Python";

export default defineConfig(({ mode }) => {
  // Prefix "" so the config can read GUI_PORT and ALLOWED_HOSTS, which are not
  // VITE_ keys. This does NOT expose them to the browser bundle: what reaches
  // `import.meta.env` is governed by envPrefix, which stays at its VITE_
  // default. Interpreters, weights and MEDIA_ROOTS remain server-side.
  const env = loadEnv(mode, ENV_DIR, "");

  // The relay is proxied rather than called cross-origin, so the browser stays
  // on one origin: no CORS preflight on the artifact images, and no certificate
  // warning when the relay is running with self-signed TLS.
  //
  // Derived from GUI_PORT rather than hardcoded, because GUI_PORT is
  // configurable in that same .env — a relay moved to another port with a
  // proxy still pointed at 7100 gives a console whose every pane is empty and
  // whose only symptom is a connection refused in a terminal nobody is reading.
  const relayPort = env.GUI_PORT || "7100";
  const RELAY = env.VITE_RELAY ?? process.env.VITE_RELAY ?? `http://127.0.0.1:${relayPort}`;

  /**
   * Hostnames this dev server will answer to.
   *
   * Vite refuses requests whose Host header it does not recognise — a
   * DNS-rebinding defence, and it is on by default. IP addresses and localhost
   * are always allowed, so reaching the console at http://<ip>:5273 works with
   * no configuration. A NAME does not: http://dgx-spark.local:5273 gets
   * "Blocked request. This host is not allowed." and nothing else, which reads
   * like the server is down.
   *
   * So list the names the box answers to:
   *
   *     ALLOWED_HOSTS=dgx-spark.local,dgx-spark
   *
   * or ALLOWED_HOSTS=all on a closed network to switch the check off entirely.
   * Left unset it stays at Vite's default, which is the safe one.
   */
  const hosts = (env.ALLOWED_HOSTS ?? "").trim();
  const allowedHosts =
    hosts === "all" || hosts === "true"
      ? true
      : hosts
        ? hosts.split(",").map((h) => h.trim()).filter(Boolean)
        : undefined;

  return {
    plugins: [react()],
    envDir: ENV_DIR,

    server: {
      host: true,
      port: 5273,
      ...(allowedHosts === undefined ? {} : { allowedHosts }),
      proxy: {
        "/socket.io": { target: RELAY, ws: true, changeOrigin: true, secure: false },
        "/artifacts": { target: RELAY, changeOrigin: true, secure: false },
        "/health": { target: RELAY, changeOrigin: true, secure: false },
        // The launcher's file picker and its readiness probe.
        "/media": { target: RELAY, changeOrigin: true, secure: false },
        "/launcher": { target: RELAY, changeOrigin: true, secure: false },
      },
    },
  };
});
