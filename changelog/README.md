# CHANGELOG

This directory contains "news fragments",
i.e. short files that contain a small markdown-formatted bit of text
that will be added to the CHANGELOG when it is next compiled.

The CHANGELOG will be read by users,
so this description should be aimed to Expert prior elicitation method users
instead of describing internal changes which are only relevant to developers.
Merge requests in combination with our git history
provide additional developer-centric information.

Make sure to use phrases in the past tense and use punctuation, examples:

```
Improved verbose diff output with sequences.

Terminal summary statistics now use multiple colors.
```

Each file should have a name of the form `<MR>.<TYPE>.md`, where `<MR>` is the merge request number, and `<TYPE>` is one of:

* `feature`: new user facing features, like new command-line options and new behaviour.
* `improvement`: improvement of existing functionality, usually without requiring user intervention
* `fix`: fixes a bug.
* `docs`: documentation improvement, like rewording an entire section or adding missing docs.
* `deprecation`: feature deprecation.
* `breaking`: a change which may break existing uses, such as feature removal or behaviour change.
* `trivial`: fixing a small typo or internal change that might be noteworthy.

So for example: `123.feature.md`, `456.fix.md`.

Since you need the merge request number for the filename, you must submit a MR first.
From this MR, you can get the MR number and then create the news file.
A single MR can also have multiple news items,
for example a given MR may add a feature as well as deprecate some existing functionality.

If you are not sure what issue type to use, don't hesitate to ask in your MR.

`towncrier` preserves multiple paragraphs and formatting
(code blocks, lists, and so on),
but for entries other than features it is usually better to stick to a single paragraph to keep it concise.
You may also use [mkdocs-autorefs cross-referencing](https://mkdocstrings.github.io/autorefs/)
within your news items to link to other documentation.

You can also run `towncrier build --draft --version draft`
to see the draft changelog that will be appended to the CHANGELOG on the next release.
