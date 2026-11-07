# pbrAudioRender
This release introduces significant updates to the `pbrAudioRender` ecosystem, focusing on performance, improved storage backends, and UI enhancements.

**Breaking Changes:**
* This version includes a complete rewrite of the `pbrAudioStorage` engine. Users relying on specific metadata structures or backend storage may need to review their current implementations.

**Key Highlights:**
* **New UI/UX:** `pbrAudioBlender` now features a 3D viewport widget GUI and a dedicated Viewport Panel. A TUI-based `pbrAudioDAW` has also been added.
* **Performance:** `pbrAudioStorage` has been refactored to support parallel appending and memory-mapped files, significantly improving handling of large data sets.
* **New Processors:** Added support for advanced modal displacement and acoustic pressure processing.
* **Storage Improvements:** Improved metadata handling and compression support (BZ2) for saved entities.
* **Bug Fixes:** Resolved critical issues, including memory exhaustion in `ModalComposer` and various import/naming errors throughout the integration layers.
