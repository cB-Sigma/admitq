# Security policy

## Supported versions

admitq is pre-1.0.  Security fixes are made on the latest release only.

| Version | Supported |
|---------|-----------|
| 0.1.x   | yes       |

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Report it privately through GitHub:
<https://github.com/cb-Sigma/admitq/security/advisories/new>

Include what you can of: the affected version, a description of the problem, how to
reproduce it and what an attacker gains.  You should get an acknowledgement within a
week.  Once a fix is ready we will publish a release and an advisory, and credit you
unless you prefer otherwise.

## What counts

admitq sits on the request path and makes allow/deny decisions, so we are interested in
anything that lets a client:

- bypass a budget or an in-flight limit it should be subject to;
- spend another tenant's budget or starve another tenant of capacity;
- make the middleware hold capacity forever (a leak) or consume unbounded memory;
- crash the middleware with a crafted request or response body.

## Things to know when deploying

These are properties of the design, not vulnerabilities - but they matter:

- **Tenant identity is whatever your key function returns.**  `header_tenant` and
  `header_priority` trust a request header; use them only for headers set or verified by
  your own edge.  `jwt_claim_tenant` decodes a JWT **without verifying it** and is only
  safe behind something that has already verified the signature.
- **The default tenant key is a hash of the API key.**  admitq does not authenticate
  anyone; an unknown key is simply a new tenant with a fresh default budget.  Put
  authentication in front of it.
- **Cost estimates are not a security boundary.**  A client can make the pre-admission
  estimate too low, for example with text the character heuristic under-counts.  Budgets
  are corrected with real usage after the fact and debt is bounded at one bucket, so the
  gain is limited, but an under-estimated request still holds less in-flight capacity
  than it should while it runs.
- **The metrics endpoint is unauthenticated.**  `metrics_path` exposes tenant labels
  (hashed keys by default) and traffic volumes.  Keep it on an internal listener or behind
  your usual scrape authentication.
- **Redis keys contain tenant names.**  With `RedisBackend`, protect Redis the way you
  protect any datastore: a client that can write to it can reset or drain budgets.
