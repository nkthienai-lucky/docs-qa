# Repository conventions

## Agent and planning artifacts

Never commit or push local agent/planning scratch files.

- `.superpowers/` (including `sdd/` workspaces, progress ledgers, task briefs,
  logs and review artifacts) is local-only scratch. It must not be tracked.
- `docs/superpowers/` (plans, specs, audits and similar agent-generated design
  documents) is local-only working material. It must not be tracked or pushed.
- `.claude/` is user-owned local state. Leave it untouched and untracked.
- Do not add these paths with `git add -f`. If a rule or hook tries to stage or
  commit them, remove them from the index.

Technical decisions and operational constraints that must survive in the
repository belong in project documentation such as `README.md` or
`docs/corpus.md`, not in agent planning artifacts.

## Git and remote

- Keep `origin` on SSH (`git@github.com:nkthienai-lucky/docs-qa.git`).
- Do not create feature branches for this project's work unless explicitly
  asked; commit to and push `main` directly when the user authorizes it.
- Do not push local agent scratch files under any branch.

## Runtime and deployment

- Runtime API credentials and provider details come from deployment environment
  variables only. Never commit secrets, `.env` files, provider URLs with tokens,
  or corpus excerpts.
- Public deployment is not authorized by repository changes. Auth, rate limits,
  CORS, quotas, provider retention/budget, persistent storage backups and parser
  resources remain operator gates.
- Prefer native Zed subagents for substantive code review when available; use
  them read-only unless explicitly asked to edit.
