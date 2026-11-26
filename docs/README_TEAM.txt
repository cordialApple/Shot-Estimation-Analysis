SHOT STUDY - TEAM RUN KIT
=========================

Goal: run the two foundation models (TabFM, TabICL) on free Google Colab GPUs,
one model x one direction per person, all at the same time.

Who runs what (one notebook each)
---------------------------------
  Person 1   tabfm_S1_to_S2.ipynb
  Person 2   tabfm_S2_to_S1.ipynb
  Person 3   tabicl_S1_to_S2.ipynb
  Person 4   tabicl_S2_to_S1.ipynb
  Brandon    logistic regression + gradient boosting (local), final comparison

Steps (about 5 minutes of your time; then the GPU works)
--------------------------------------------------------
1. Brandon shares the Drive folder "shotlab" with you as EDITOR.
2. In Google Drive: Shared with me > right-click "shotlab" > Organize >
   Add shortcut > My Drive. The folder must appear in your My Drive with the
   name "shotlab" (exactly).
3. Open the folder, double-click YOUR notebook, choose "Open with Google
   Colaboratory".
4. Runtime > Change runtime type > T4 GPU > Save.
5. Runtime > Run all.
   - Approve "Connect to Google Drive" with the same Google account and allow
     all the requested permissions.
6. The first run installs pinned packages and restarts the runtime once.
   A "session crashed" notice at that point is expected.
   Click Runtime > Run all again.
7. Leave the tab open and the computer awake. Progress lines print under
   "3. Primary run".
8. If Colab disconnects or hits a usage limit: reconnect (or come back later)
   and Run all again. Finished work is skipped; at most one 2,000-row chunk
   is lost.
9. Finished when the last cell prints a line with results_<your part>.zip.
   Nothing to send: everything is already in the shared folder. Tell Brandon.

Rules
-----
- Do not edit code cells. Do not open or run another person's notebook.
- Do not rename, move or delete anything in the shared folder.
- If you are short on time: the important part ends with section
  "4. Explanation probes". Section "5. Side arms" is long and can be stopped;
  whatever finished is kept.

What the notebook does
----------------------
- Timing pilot on training-season rows (skipped if a teammate on the same
  model already ran it; the table is shared).
- Picks the grid tier that fits 2 GPU-hours of tuning (rule fixed in advance,
  docs/freeze.md entry 15). You do not choose anything.
- Tunes on the training season, calibrates, predicts the 50,000-shot panel of
  the other season with three context seeds.
- Computes NO held-out score. The panel files have no outcome column. Scoring
  happens later on one machine, once, for all four models.

If you cannot use the shared folder
-----------------------------------
Make a folder named "shotlab" in your own My Drive, copy into it your notebook
and either shotlab_bundle.zip or all shotlab_bundle.zip.partNN files, run the
same steps, then send back results_<your part>.zip.

Notes
-----
- TabFM weights are released for non-commercial use only.
- Bundle: shotlab_bundle.zip, sha256
  37bc303f7829f5198548178fab2013b2f479da54188aef8cfd076770333ff4a2
