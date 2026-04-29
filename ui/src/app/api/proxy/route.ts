import { type NextRequest, NextResponse } from "next/server";

const API_URL = process.env.CORTEX_API_URL ?? "http://localhost:8000";

export async function GET(request: NextRequest): Promise<Response> {
  return proxy(request);
}

export async function POST(request: NextRequest): Promise<Response> {
  return proxy(request);
}

async function proxy(request: NextRequest): Promise<Response> {
  const target = request.nextUrl.searchParams.get("path") ?? "/";
  const response = await fetch(`${API_URL}${target}`, {
    method: request.method,
    headers: request.headers,
    body: request.method === "GET" ? undefined : await request.text()
  });
  return new NextResponse(response.body, {
    status: response.status,
    headers: response.headers
  });
}
