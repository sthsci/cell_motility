# Nature Communications Article preparation record

Target: **Nature Communications, Article, initial submission draft**.
Checked: **2026-10-28**. Status: **assistant-checked; human/editorial verification
pending**. This record is formatting guidance, not confirmation of editorial
suitability or submission readiness.

## Official sources and access

- [How to submit](https://www.nature.com/ncomms/submit/how-to-submit): complete
  official page content was returned by the web search service; direct opening
  redirected to Nature authentication. Relevant headings: Manuscript text,
  TeX/LaTeX, References, Figures, Funding sources.
- [Guide to formatting articles](https://www.nature.com/documents/ncomms-formatting-instructions.pdf):
  successfully opened six-page official PDF. It is currently served but bears
  **Revised May 7, 2021**; do not misrepresent this as a newly issued 2026 guide.
- [Current policy guide](https://www.nature.com/ncomms/submit/policy): official
  page content returned by the web search service; direct opening redirected.
- [Nature Portfolio AI policy](https://www.nature.com/nature-portfolio/editorial-policies/ai):
  direct opening redirected. The journal's own submission page explicitly
  states the Methods disclosure and authorship requirements.
- [Current Springer Nature AI manuscript policy](https://www.springernature.com/gp/policies/editorial-policies/ai-manuscript-preparation):
  verifiable data-derived graphics require disclosure, traceable inputs, and
  human validation. Recheck current policy directly before submission.

## LaTeX and package choice

The journal accepts standard classes, including `article.cls`, with default
Computer Modern fonts. It recommends `graphicx` and numerical citations.
Use one textual manuscript file with the rendered bibliography embedded;
an external `.bib` file is useful during drafting but does not replace that
submission requirement. Avoid custom macros and nonstandard fonts/packages.
Initial submission can combine text and figures; revised text and figure
files are handled separately. A visual imitation of the published journal
layout is unnecessary: production applies its own style.

For this project, a standard `article` manuscript and separate Supplementary
Information PDF are the appropriate targets. They must not be labeled an
official Nature-provided class or template. Keep the full draft editable,
compile locally, and inspect the output. Export original numerical plots as
vector PDF/SVG with editable text; supply figure-specific source data.
These choices follow the [submission instructions](https://www.nature.com/ncomms/submit/how-to-submit).

## Structural checks from the official formatting PDF

| Item | Draft target |
|---|---|
| Title | At most 15 words; avoid punctuation and puns |
| Abstract | At most 150 words; no references or undefined abbreviations |
| Main text | At most 6,000 words excluding Methods and figure legends |
| Main displays | At most 10 figures plus tables combined |
| Introduction and Discussion | No subheadings |
| Results and Methods | Unnumbered subheadings, at most 60 characters |
| References | Numerical, citation order; at most 70 |
| Figure legends | Standalone title and panel descriptions; at most 350 words |
| Panel labels | Lower-case bold letters; consistent symbols and units |
| Supplement | Separate self-contained PDF, starting with article title |

Use the sequence Introduction, Results, Discussion, Methods, Data Availability,
Code Availability, References and declarations. Keep data and code statements
separate. Cite supplementary items specifically (for example, Supplementary
Fig. 1). These are conservative drafting targets from the [currently served
PDF](https://www.nature.com/documents/ncomms-formatting-instructions.pdf);
initial submissions have format flexibility.

## Policy items requiring factual author input

The live [policy guide](https://www.nature.com/ncomms/submit/policy) requires
authorship contributions, competing interests, funding, data access and, where
central, code access statements. It requests enduring code deposition and
software/version information. Funding is now explicitly required, although the
older formatting PDF places funding under acknowledgements; follow the live
policy and retain a separate Funding statement.

This study contains only generated trajectories, so the draft may accurately
state that no participants, animals, biological materials or experimental
images were studied. It must not invent an ethics approval, grant, repository
DOI, authorship list, author review, or absence of conflicts. Mark unavailable
declarations as pending.

LLM assistance here includes code, analyses and drafting, so it exceeds
copyediting and should be disclosed in Methods. AI must not appear as an
author. Record what was actually done and leave human validation pending;
do not claim authors reviewed/approved generated material before they have.

## Submission-stage status

Prepare and compile the manuscript, supplement, original figures and source
data locally. No journal upload, repository publication or manuscript
submission is authorized by this drafting task. Before actual submission,
the authors need to resolve declarations, verify the evidence and provenance,
choose a durable data/code archive, review the journal's reporting-summary
requirements and inspect the submission-system rendering.
