/**
 * Files this machine's access request into Firestore, and reads back its status.
 *
 * There is no server here to write it for us — the assistant is two local
 * processes — so the client files its own request and an admin approves it from
 * the AI Calculator's "Manage access" dashboard, which runs against the same
 * Firebase project. One approval, both applications.
 *
 * What stops a user approving themselves is `firestore.rules`, which runs on
 * Google's servers: you may create only a *pending* request, only for your own
 * uid, only with your own verified email, and only an admin may change a
 * status. Nothing written here is trusted by anything.
 */

import { firebase } from "../config";

export type RequestStatus = "pending" | "approved" | "rejected";

const base = () =>
  `https://firestore.googleapis.com/v1/projects/${firebase.projectId}` +
  `/databases/(default)/documents/accessRequests`;

/** Firestore's typed-value JSON, for the one document shape we write. */
function toFields(uid: string, email: string) {
  return {
    fields: {
      uid: { stringValue: uid },
      email: { stringValue: email.toLowerCase() },
      status: { stringValue: "pending" },
      isAdmin: { booleanValue: false },
      createdAt: { timestampValue: new Date().toISOString() },
    },
  };
}

function statusOf(doc: unknown): RequestStatus | null {
  const value = (doc as { fields?: { status?: { stringValue?: string } } })?.fields
    ?.status?.stringValue;
  return value === "approved" || value === "rejected" || value === "pending"
    ? value
    : null;
}

/**
 * Returns this account's request status, creating a pending request the first
 * time. Idempotent. Returns null when Firestore can't be reached or the rules
 * refuse — callers fall back to whatever the ID token's claims say, which is
 * the authoritative answer anyway.
 */
export async function recordOwnAccessRequest(
  uid: string,
  email: string,
  idToken: string
): Promise<RequestStatus | null> {
  if (!firebase.projectId || !uid) return null;
  const headers = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${idToken}`,
  };

  try {
    const existing = await fetch(`${base()}/${encodeURIComponent(uid)}`, { headers });
    if (existing.ok) return statusOf(await existing.json());
    // Anything other than "no such document" means we can't help here.
    if (existing.status !== 404) return null;

    const created = await fetch(`${base()}?documentId=${encodeURIComponent(uid)}`, {
      method: "POST",
      headers,
      body: JSON.stringify(toFields(uid, email)),
    });
    // 409 means it appeared between our read and our write — still pending.
    if (created.ok || created.status === 409) return "pending";
    return null;
  } catch {
    /* offline, blocked, or the rules refused — leave it to the caller */
    return null;
  }
}
