import csv
import io
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

import pipeline

st.set_page_config(
    page_title="Signature check",
    page_icon=":material/verified_user:",
    layout="wide",
)

st.title("Signature check")
st.caption("Compare customer letters with the master signature register.")


def _encode_signature_png(image):
    if image is None:
        return b""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode a signature crop as PNG.")
    return encoded.tobytes()


def _build_results_csv(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=pipeline.COLUMNS)
    writer.writeheader()
    for row in rows:
        csv_row = {field: row.get(field, "") for field in pipeline.COLUMNS}
        writer.writerow(csv_row)
    return output.getvalue().encode("utf-8")


def _style_result(result):
    styles = {
        "MATCH": "background-color: #d1fae5; color: #065f46; font-weight: 600",
        "MISMATCH": "background-color: #fee2e2; color: #991b1b; font-weight: 600",
        "NOT_FOUND": "background-color: #fee2e2; color: #991b1b; font-weight: 600",
        "REVIEW": "background-color: #fef3c7; color: #92400e; font-weight: 600",
    }
    return styles.get(result, "")


_VERDICT_BADGE_COLORS = {
    "MATCH": "green",
    "MISMATCH": "red",
    "NOT_FOUND": "red",
    "REVIEW": "yellow",
}


def _reset_uploads():
    st.session_state["upload_generation"] = st.session_state.get("upload_generation", 0) + 1
    st.session_state.pop("signature_results", None)


upload_generation = st.session_state.setdefault("upload_generation", 0)

scope = st.segmented_control(
    "Verification scope",
    options=["All letters", "One customer"],
    default="All letters",
    key="verification_scope",
)

with st.container(border=True):
    st.subheader("Source documents")
    master_upload = st.file_uploader(
        "Master signature register",
        type=["pdf"],
        help="Upload the PDF containing the reference signatures.",
        key=f"master_pdf_{upload_generation}",
    )
    letter_uploads = st.file_uploader(
        "Customer letters",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more customer letter PDFs.",
        key=f"customer_letters_{upload_generation}",
    )

    selected_upload = None
    if scope == "One customer" and letter_uploads:
        selected_index = st.selectbox(
            "Customer letter to check",
            options=range(len(letter_uploads)),
            format_func=lambda index: f"{letter_uploads[index].name} ({index + 1})",
        )
        selected_upload = letter_uploads[selected_index]

    uploads_ready = bool(master_upload and letter_uploads)
    if scope == "One customer":
        uploads_ready = uploads_ready and selected_upload is not None

    run_check = st.button(
        "Run verification",
        type="primary",
        icon=":material/fingerprint:",
        disabled=not uploads_ready,
    )
    st.button(
        "Reset",
        icon=":material/refresh:",
        on_click=_reset_uploads,
    )

if run_check:
    selected_uploads = letter_uploads if scope == "All letters" else [selected_upload]
    try:
        with tempfile.TemporaryDirectory(prefix="signature_check_") as temporary_dir:
            temp_path = Path(temporary_dir)
            master_path = temp_path / "master_register.pdf"
            master_path.write_bytes(master_upload.getvalue())

            with st.status(
                f"Checking {len(selected_uploads)} letter(s)...", expanded=True
            ) as status:
                store = pipeline.build_master_store(master_path)
                if not store:
                    raise ValueError("No master signatures could be extracted from the PDF.")

                rows = []
                progress = st.progress(0, text="Preparing signature comparisons")
                for index, uploaded_letter in enumerate(selected_uploads):
                    safe_name = Path(uploaded_letter.name).name
                    letter_path = temp_path / f"{index:03d}_{safe_name}"
                    letter_path.write_bytes(uploaded_letter.getvalue())
                    row = pipeline.verify_letter(letter_path, store, include_images=True)
                    row["letter_file"] = uploaded_letter.name
                    row["_master_signature_png"] = _encode_signature_png(
                        row.pop("_master_signature_image")
                    )
                    row["_letter_signature_png"] = _encode_signature_png(
                        row.pop("_letter_signature_image")
                    )
                    rows.append(row)
                    progress.progress(
                        (index + 1) / len(selected_uploads),
                        text=f"Checked {uploaded_letter.name}",
                    )

                st.session_state["signature_results"] = {
                    "rows": rows,
                    "csv": _build_results_csv(rows),
                    "scope": scope,
                    "master_count": len(store),
                }
                status.update(
                    label=f"Checked {len(rows)} letter(s)",
                    state="complete",
                    expanded=False,
                )
    except Exception as exc:
        st.error(f"Verification failed: {exc}")

result = st.session_state.get("signature_results")
if result:
    rows = result["rows"]
    st.subheader("Results")
    st.caption(
        f"{result['scope']} | {result['master_count']} master signatures loaded | "
        f"{len(rows)} letter(s) checked"
    )

    counts = {
        verdict: sum(row["result"] == verdict for row in rows)
        for verdict in ("MATCH", "MISMATCH", "REVIEW", "NOT_FOUND")
    }
    metric_columns = st.columns(4)
    for column, verdict in zip(metric_columns, counts):
        column.metric(verdict.replace("_", " "), counts[verdict])

    display_rows = pd.DataFrame(
        [{field: row.get(field, "") for field in pipeline.COLUMNS} for row in rows]
    ).rename(
        columns={
            "cif": "CIF",
            "name": "Customer",
            "letter_file": "Letter",
            "similarity_score": "Score",
            "result": "Result",
            "remarks": "Remarks",
        }
    )
    display_rows["Score"] = pd.to_numeric(display_rows["Score"], errors="coerce")
    st.dataframe(
        display_rows.style.map(_style_result, subset=["Result"]),
        hide_index=True,
        column_config={"Score": st.column_config.NumberColumn(format="%.4f")},
    )
    st.subheader("Signature comparison")
    for row in rows:
        customer = row["name"] or row["cif"] or row["letter_file"]
        with st.container(border=True):
            st.markdown(f"**{customer}**")
            st.badge(
                row["result"].replace("_", " "),
                color=_VERDICT_BADGE_COLORS.get(row["result"], "gray"),
            )
            master_column, letter_column = st.columns(2)
            with master_column:
                st.caption("Master signature")
                if row["_master_signature_png"]:
                    st.image(row["_master_signature_png"])
                else:
                    st.info("Master signature unavailable")
            with letter_column:
                st.caption("Letter signature")
                if row["_letter_signature_png"]:
                    st.image(row["_letter_signature_png"])
                else:
                    st.info("Letter signature unavailable")
    st.download_button(
        "Download CSV",
        data=result["csv"],
        file_name="signature_results.csv",
        mime="text/csv",
        icon=":material/download:",
    )