# Screenshot and Demo Preparation

This folder is reserved for final application screenshots and the optional demo GIF used in the project README and portfolio documentation.

Required screenshots:

1. `home.png` - Streamlit landing view with project header and sidebar.
2. `prediction_form.png` - Applicant input form with example applicant controls.
3. `prediction_result.png` - Risk assessment result card and download center.
4. `model_information.png` - Model metrics and explainability view.
5. `about_project.png` - Project background, limitations, and report downloads.
6. `demo.gif` - short demo showing open app, select applicant, predict, and view result.

Capture screenshots after the final Streamlit smoke test:

```bash
streamlit run app/app.py
```

Do not include real customer data in screenshots. Use the built-in example applicants only.

Recommended capture flow:

1. Launch the app.
2. Capture the home/prediction landing state.
3. Select one built-in example applicant.
4. Capture the filled prediction form.
5. Submit the form and capture the result card.
6. Open Model Information and capture the explainability section.
7. Open About Project and capture the pipeline/summary section.
