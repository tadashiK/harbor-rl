<!-- Keep the change scoped to one concern. -->

## What this changes

<!-- One or two sentences. If it alters behaviour an agent depends on, say so explicitly:
     agent instructions are load-bearing here in a way ordinary comments are not. -->

## Why

<!-- What was broken or missing. Link the issue if there is one. -->

## Verification

<!-- How you know it works. For a plugin change, name the gate or test that proves it. -->

- [ ] `pytest tests/contract tests/unit` passes
- [ ] `python3 tools/gen_docs_reference.py` run, if any command or agent frontmatter changed
- [ ] New scripts have a unit test (`UNTESTED_DEBT` may only shrink)
- [ ] Generated receipts stay English-only
