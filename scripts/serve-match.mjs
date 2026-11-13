import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const types = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
};
const port = Number(process.env.PORT ?? "4173");
if (!Number.isInteger(port) || port < 0 || port > 65535) throw new RangeError("PORT must be an integer from 0 to 65535");

const server = createServer(async (request, response) => {
  try {
    if (request.method !== "GET" && request.method !== "HEAD") {
      response.writeHead(405, { Allow: "GET, HEAD" }).end();
      return;
    }
    const pathname = decodeURIComponent(new URL(request.url ?? "/", "http://127.0.0.1").pathname);
    const relative = pathname === "/" ? "browser/index.html" : pathname.slice(1);
    const file = resolve(root, relative);
    const allowed = file === resolve(root, "browser/index.html") ||
      file === resolve(root, "browser/match.css") || file.startsWith(resolve(root, "dist") + sep);
    const type = types[extname(file)];
    if (!allowed || !type || !(await stat(file)).isFile()) {
      response.writeHead(404).end("Not found");
      return;
    }
    response.writeHead(200, { "Content-Type": type, "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff" });
    if (request.method === "HEAD") response.end();
    else createReadStream(file).on("error", () => response.destroy()).pipe(response);
  } catch {
    if (!response.headersSent) response.writeHead(404).end("Not found");
    else response.destroy();
  }
});
server.listen(port, "127.0.0.1", () => {
  console.log(`Touchline Football is running at http://127.0.0.1:${server.address().port}/`);
});
