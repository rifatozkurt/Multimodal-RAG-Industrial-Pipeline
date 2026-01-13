from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
from .config import documents_path, chunking_size, chunking_step
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pymupdf  
from langchain_core.documents import Document


fitz = pymupdf

logger = logging.getLogger(__name__)


class PdfExtractionLoader:
    """
    Loader that integrates PDF-Extract-Kit layout detection outputs
    with the original PDFs and produces LangChain Documents for
    text regions and image regions.
    """

    def __init__(
        self,
        pdfs_dir: Path | str,
        layout_root_dir: Path | str,
        images_root: Optional[Path | str] = None,
        text_categories: Optional[List[str]] = None,
        image_categories: Optional[List[str]] = None,
        score_threshold: float = 0.0,
        layout_filename: str = "layout_results.json",
        image_zoom: float = 3.0,  # controls output resolution
        min_image_width: int = 50,    # skip images smaller than this width
        min_image_height: int = 50,   # skip images smaller than this height
        min_image_area: int = 2500,   # skip images smaller than this area (width * height)
    ) -> None:
        self.pdfs_dir = Path(pdfs_dir)
        self.layout_root_dir = Path(layout_root_dir)
        self.images_root = Path(images_root) if images_root is not None else None
        self.score_threshold = score_threshold
        self.layout_filename = layout_filename
        self.image_zoom = image_zoom
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height
        self.min_image_area = min_image_area

        # Default category hints if none provided
        self.text_categories = (
            [
                "text",
                "plain text",
                "title",
                "heading",
                "subtitle",
                "paragraph",
                "caption",
                "figure_caption",
            ]
            if text_categories is None
            else text_categories
        )

        self.image_categories = (
            [
                "figure",
                "image",
                "table",
                "chart",
                "graph",
                "plot",
                "diagram",
            ]
            if image_categories is None
            else image_categories
        )

    def load(self) -> Tuple[List[Document], List[Document]]:
        text_docs: List[Document] = []
        image_docs: List[Document] = []

        if not self.pdfs_dir.is_dir():
            raise ValueError(f"pdfs_dir is not a directory: {self.pdfs_dir}")
        if not self.layout_root_dir.is_dir():
            raise ValueError(f"layout_root_dir is not a directory: {self.layout_root_dir}")

        pdf_paths = sorted(self.pdfs_dir.glob("*.pdf"))
        if not pdf_paths:
            logger.warning("No PDF files found in %s", self.pdfs_dir)

        logger.info("Extracting layout images from PDFs...")

        for pdf_path in pdf_paths:
            pdf_name = pdf_path.name
            pdf_stem = pdf_path.stem
            layout_path = self._find_layout_json(pdf_stem)

            if layout_path is None:
                logger.warning(
                    "No layout JSON found for PDF '%s' (expected under %s)",
                    pdf_name,
                    self.layout_root_dir,
                )
                continue

            logger.info("Processing PDF '%s' with layout '%s'", pdf_path, layout_path)

            try:
                with layout_path.open("r", encoding="utf-8") as f:
                    layout_data = json.load(f)
            except Exception as e:
                logger.error("Failed to read layout JSON '%s': %s", layout_path, e)
                continue

            try:
                text_docs_pdf, image_docs_pdf = self._process_single_pdf(
                    pdf_path=pdf_path,
                    pdf_name=pdf_name,
                    pdf_stem=pdf_stem,
                    layout_data=layout_data,
                )
            except Exception as e:
                logger.exception("Error while processing PDF '%s': %s", pdf_path, e)
                continue

            text_docs.extend(text_docs_pdf)
            image_docs.extend(image_docs_pdf)

        logger.info("Loaded %d layout-image documents from PDFs.", len(image_docs))
        return text_docs, image_docs

    def _find_layout_json(self, pdf_stem: str) -> Optional[Path]:
        candidate = self.layout_root_dir / pdf_stem / self.layout_filename
        return candidate if candidate.is_file() else None

    def _process_single_pdf(
        self,
        pdf_path: Path,
        pdf_name: str,
        pdf_stem: str,
        layout_data: List[Dict[str, Any]],
    ) -> Tuple[List[Document], List[Document]]:
        text_docs: List[Document] = []
        image_docs: List[Document] = []

        with fitz.open(pdf_path) as doc:
            for page_entry in layout_data:
                page_info = page_entry.get("page_info", {})
                page_no = page_info.get("page_no")  # 0-based

                if page_no is None:
                    logger.warning(
                        "Missing 'page_no' in layout entry for '%s'; skipping page entry",
                        pdf_name,
                    )
                    continue

                if not (0 <= page_no < len(doc)):
                    logger.warning(
                        "Invalid page_no=%s for PDF '%s' (page_count=%s); skipping",
                        page_no,
                        pdf_name,
                        len(doc),
                    )
                    continue

                page = doc[page_no]
                page_num = page_no + 1  # 1-based
                page_rect = page.rect

                layout_width = float(page_info.get("width", page_rect.width))
                layout_height = float(page_info.get("height", page_rect.height))

                page_text_full = page.get_text() or ""
                layout_dets = page_entry.get("layout_dets", [])
                if not layout_dets:
                    continue

                for det_idx, det in enumerate(layout_dets):
                    score = float(det.get("score", 1.0))
                    if score < self.score_threshold:
                        continue

                    category_type = str(det.get("category_type", "")).strip()
                    category_id = det.get("category_id")
                    poly = det.get("poly")

                    rect = self._poly_to_rect(
                        poly=poly,
                        page_rect=page_rect,
                        layout_width=layout_width,
                        layout_height=layout_height,
                    )
                    if rect is None:
                        logger.debug(
                            "Skipping detection with invalid rect on page %s of '%s'",
                            page_no,
                            pdf_name,
                        )
                        continue

                    lower_cat = category_type.lower()

                    if self._is_text_category(lower_cat):
                        text_doc = self._build_text_document(
                            page=page,
                            pdf_path=pdf_path,
                            pdf_name=pdf_name,
                            pdf_stem=pdf_stem,
                            page_num=page_num,
                            page_rect=page_rect,
                            rect=rect,
                            category_type=category_type,
                            category_id=category_id,
                            poly=poly,
                            score=score,
                            page_text_full=page_text_full,
                        )
                        if text_doc is not None:
                            text_docs.append(text_doc)

                    elif self._is_image_category(lower_cat):
                        image_doc = self._build_image_document(
                            page=page,
                            pdf_path=pdf_path,
                            pdf_name=pdf_name,
                            pdf_stem=pdf_stem,
                            page_num=page_num,
                            page_rect=page_rect,
                            rect=rect,
                            category_type=category_type,
                            category_id=category_id,
                            poly=poly,
                            score=score,
                            det_idx=det_idx,
                            page_text_full=page_text_full,
                        )
                        if image_doc is not None:
                            image_docs.append(image_doc)

                    else:
                        logger.debug(
                            "Skipping detection with category '%s' on page %s of '%s'",
                            category_type,
                            page_no,
                            pdf_name,
                        )

        return text_docs, image_docs


    def _is_text_category(self, lower_cat: str) -> bool:
        if "caption" in lower_cat:
            return True
        return any(token in lower_cat for token in self.text_categories)

    def _is_image_category(self, lower_cat: str) -> bool:
        if "caption" in lower_cat:
            return False
        return any(token in lower_cat for token in self.image_categories)


    @staticmethod
    def _poly_to_rect(
        poly: Any,
        page_rect: fitz.Rect,
        layout_width: float,
        layout_height: float,
    ) -> Optional[fitz.Rect]:
        if not isinstance(poly, (list, tuple)) or len(poly) < 8 or len(poly) % 2 != 0:
            return None

        xs = poly[0::2]
        ys = poly[1::2]

        try:
            if layout_width > 0 and layout_height > 0:
                sx = page_rect.width / layout_width
                sy = page_rect.height / layout_height

                pdf_xs = []
                pdf_ys = []
                for x, y in zip(xs, ys):
                    x_pdf = page_rect.x0 + float(x) * sx
                    y_pdf = page_rect.y0 + float(y) * sy  # no flip
                    pdf_xs.append(x_pdf)
                    pdf_ys.append(y_pdf)
            else:
                pdf_xs = [float(x) for x in xs]
                pdf_ys = [float(y) for y in ys]

            x_min = max(page_rect.x0, min(pdf_xs))
            x_max = min(page_rect.x1, max(pdf_xs))
            y_min = max(page_rect.y0, min(pdf_ys))
            y_max = min(page_rect.y1, max(pdf_ys))

        except Exception:
            return None

        if x_min >= x_max or y_min >= y_max:
            return None

        return fitz.Rect(x_min, y_min, x_max, y_max)


    def _build_text_document(
        self,
        page: fitz.Page,
        pdf_path: Path,
        pdf_name: str,
        pdf_stem: str,
        page_num: int,
        page_rect: fitz.Rect,
        rect: fitz.Rect,
        category_type: str,
        category_id: Any,
        poly: Any,
        score: float,
        page_text_full: str,
    ) -> Optional[Document]:
        region_text = page.get_textbox(rect) or ""
        region_text = region_text.strip()

        if not region_text:
            region_text = page.get_text("text", clip=rect).strip()

        if not region_text:
            return None

        metadata: Dict[str, Any] = {
            "type": "layout_text",
            "doc_id": pdf_name,
            "file_path": str(pdf_path),
            "page": page_num,
            "page_width": page_rect.width,
            "page_height": page_rect.height,
            "bbox_page_coords": str(
                (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
            ),
            "bbox": str((float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))),
            "poly": str(poly),
            "category_type": category_type,
            "category_id": category_id,
            "score": score,
            "has_image": False,
            "page_text": page_text_full,
            "region_text": region_text,
            "url": f"{pdf_name}#page={page_num}",
            "image_path": "",
            "image_ext": "",
            "width": 0,
            "height": 0,
        }

        return Document(page_content=region_text, metadata=metadata)

    def _build_image_document(
        self,
        page: fitz.Page,
        pdf_path: Path,
        pdf_name: str,
        pdf_stem: str,
        page_num: int,
        page_rect: fitz.Rect,
        rect: fitz.Rect,
        category_type: str,
        category_id: Any,
        poly: Any,
        score: float,
        det_idx: int,
        page_text_full: str,
    ) -> Optional[Document]:
        if self.images_root is None:
            logger.debug(
                "images_root is None -> skipping image region on page %s of '%s'",
                page_num,
                pdf_name,
            )
            return None

        rel_dir = Path(pdf_stem)
        rel_name = f"{pdf_stem}_p{page_num}_det{det_idx}.png"
        rel_path = rel_dir / rel_name
        full_path = self.images_root / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # HIGHER QUALITY RENDERING HERE
            mat = fitz.Matrix(self.image_zoom, self.image_zoom)
            pix = page.get_pixmap(matrix=mat, clip=rect)
            width, height = pix.width, pix.height
            if width <= 0 or height <= 0:
                logger.warning(
                    "Skipping zero-size crop for '%s' (page %s, det %s)",
                    pdf_name,
                    page_num,
                    det_idx,
                )
                return None
            
            # Check size thresholds
            if width < self.min_image_width or height < self.min_image_height:
                logger.debug(
                    "Skipping small image crop for '%s' (page %s, det %s): %dx%d < %dx%d",
                    pdf_name,
                    page_num,
                    det_idx,
                    width,
                    height,
                    self.min_image_width,
                    self.min_image_height,
                )
                return None
            
            if width * height < self.min_image_area:
                logger.debug(
                    "Skipping small area image crop for '%s' (page %s, det %s): area %d < %d",
                    pdf_name,
                    page_num,
                    det_idx,
                    width * height,
                    self.min_image_area,
                )
                return None
            
            pix.save(full_path.as_posix())
        except Exception as e:
            logger.error(
                "Failed to generate image crop for '%s' (page %s, det %s): %s",
                pdf_name,
                page_num,
                det_idx,
                e,
            )
            return None

        metadata: Dict[str, Any] = {
            "type": "layout_image",
            "doc_id": pdf_name,
            "file_path": str(pdf_path),
            "page": page_num,
            "page_width": page_rect.width,
            "page_height": page_rect.height,
            "bbox_page_coords": str(
                (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
            ),
            "bbox": str((float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))),
            "poly": str(poly),
            "category_type": category_type,
            "category_id": category_id,
            "score": score,
            "image_path": str(full_path),
            "image_rel_path": str(rel_path),
            "image_ext": "png",
            "width": width,
            "height": height,
            "has_image": True,
            "page_text": page_text_full,
            "url": f"{pdf_name}#page={page_num}",
        }

        return Document(page_content=str(full_path), metadata=metadata)

class PdfImagesLoader:
    """
    Extracts images from a PDF and returns one LangChain Document per image.
    Optionally saves each page as image and crops pages into a grid.

    Returns from .load():
        image_docs: list[Document]      # embedded images
        page_docs:  list[Document]      # full-page images + cropped page tiles
    """
    def __init__(
        self,
        pdf_path: str,
        images_root_dir: str,
        min_dim: int = 50,            # skip tiny images
        min_bytes: int = 0,           # skip small files
        min_bytes_per_px: float = 0,  # filter very compressed images if > 0
        extract_page_text: bool = True,
        store_to_folder: bool = True,
        pages_as_images: bool = False,
        pages_root_dir: str | None = None,
        page_zoom: float = 2.0,       # 1.0 ≈ 72dpi, 2.0 ≈ ~144dpi
        crop_pages: tuple[int, int] | None = None,  # (num_cols, num_rows)
    ):
        self.pdf_path = pdf_path
        self.images_root_dir = images_root_dir
        self.min_dim = min_dim
        self.min_bytes = min_bytes
        self.min_bytes_per_px = min_bytes_per_px
        self.extract_page_text = extract_page_text
        self.store_to_folder = store_to_folder

        self.pages_as_images = pages_as_images
        self.pages_root_dir = pages_root_dir
        self.page_zoom = page_zoom

        # crop_pages = (x_cols, y_rows); if None or (1,1) => no cropping
        self.crop_pages = crop_pages

    # this function is not my own, taken from pymupdf docs
    @staticmethod
    def _recover_pix(doc, xref, smask):
        if smask and smask > 0:
            base = pymupdf.Pixmap(doc.extract_image(xref)["image"])
            if base.alpha:
                base = pymupdf.Pixmap(base, 0)
            mask = pymupdf.Pixmap(doc.extract_image(smask)["image"])
            try:
                pix = pymupdf.Pixmap(base, mask)
            except Exception:
                pix = pymupdf.Pixmap(doc.extract_image(xref)["image"])
            ext = "pam" if (getattr(base, "n", 3) > 3) else "png"
            return {
                "ext": ext,
                "colorspace": pix.colorspace.n if pix.colorspace else 3,
                "image": pix.tobytes(ext),
            }

        if "/ColorSpace" in doc.xref_object(xref, compressed=True):
            pix = pymupdf.Pixmap(doc, xref)
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            return {"ext": "png", "colorspace": 3, "image": pix.tobytes("png")}

        return doc.extract_image(xref)

    def _render_page(self, page, out_dir_pages, pdf_stem, pdf_name, page_text):
        """Render a single full page to PNG and return a Document."""
        page_num = page.number + 1

        mat = pymupdf.Matrix(self.page_zoom, self.page_zoom)
        pix = page.get_pixmap(matrix=mat)  # RGB
        width, height = pix.width, pix.height
        filename = f"{pdf_stem}_p{page_num}_w{width}_h{height}.png"
        save_path = os.path.join(out_dir_pages, filename)

        if self.store_to_folder:
            os.makedirs(out_dir_pages, exist_ok=True)
            pix.save(save_path)

        return Document(
            page_content="",
            metadata={
                "type": "page_image",
                "doc_id": pdf_name,
                # add path of original pdf
                "file_path": self.pdf_path,
                "page": page_num,
                "page_image_path": save_path,
                "image_ext": "png",
                "width": width,
                "height": height,
                "has_image": True,
                "page_text": page_text,
                "url": f"{pdf_name}#page={page_num}",
            },
        )

    def _render_page_crops(self, page, out_dir_pages, pdf_stem, pdf_name, page_text):
        """
        Divide the page into a (cols x rows) grid and render each tile as an image.
        Returns a list of Documents with type == "page_crop".
        """
        if not self.crop_pages or self.crop_pages == (1, 1):
            return []

        num_cols, num_rows = self.crop_pages
        if num_cols <= 0 or num_rows <= 0:
            return []

        page_num = page.number + 1
        mat = pymupdf.Matrix(self.page_zoom, self.page_zoom)

        # Page coordinates (points)
        rect = page.rect
        page_w = rect.width
        page_h = rect.height

        cell_w = page_w / num_cols
        cell_h = page_h / num_rows

        crop_docs = []
        os.makedirs(out_dir_pages, exist_ok=True)

        for row in range(num_rows):
            for col in range(num_cols):
                # Define crop rectangle in page coordinates
                x0 = rect.x0 + col * cell_w
                y0 = rect.y0 + row * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h
                crop_rect = pymupdf.Rect(x0, y0, x1, y1)

                # Render only this region
                pix = page.get_pixmap(matrix=mat, clip=crop_rect)
                width, height = pix.width, pix.height

                filename = (
                    f"{pdf_stem}_p{page_num}_r{row}_c{col}_"
                    f"w{width}_h{height}.png"
                )
                save_path = os.path.join(out_dir_pages, filename)

                if self.store_to_folder:
                    pix.save(save_path)

                crop_docs.append(
                    Document(
                        page_content="",
                        metadata={
                            "type": "page_crop",
                            "doc_id": pdf_name,
                            "page": page_num,
                            "file_path": self.pdf_path,
                            "page_image_path": save_path,
                            "image_ext": "png",
                            "width": width,
                            "height": height,
                            "has_image": True,
                            "page_text": page_text,
                            "crop_row": row,
                            "crop_col": col,
                            "grid_cols": num_cols,
                            "grid_rows": num_rows,
                            "bbox_page_coords": str((float(x0), float(y0), float(x1), float(y1))),
                            "url": f"{pdf_name}#page={page_num}",
                        },
                    )
                )

        return crop_docs

    def load(self):
        image_docs = []
        page_docs = []

        pdf_name = os.path.basename(self.pdf_path)
        pdf_stem = os.path.splitext(pdf_name)[0]

        out_dir_imgs = os.path.join(self.images_root_dir, pdf_stem)
        if self.store_to_folder and not os.path.exists(out_dir_imgs):
            os.makedirs(out_dir_imgs, exist_ok=True)

        out_dir_pages = None
        if self.pages_as_images or (self.crop_pages and self.crop_pages != (1, 1)):
            # We need a pages_dir when we either save full pages or crops
            if not self.pages_root_dir:
                raise ValueError(
                    "pages_root_dir must be provided when pages_as_images=True "
                    "or crop_pages is set."
                )
            out_dir_pages = os.path.join(self.pages_root_dir, pdf_stem)
            if self.store_to_folder and not os.path.exists(out_dir_pages):
                os.makedirs(out_dir_pages, exist_ok=True)

        with pymupdf.open(self.pdf_path) as doc:
            for page in doc:
                page_num = page.number + 1
                page_text = page.get_text() if self.extract_page_text else ""

                # --- Full page image ---
                if self.pages_as_images:
                    page_docs.append(
                        self._render_page(
                            page=page,
                            out_dir_pages=out_dir_pages,
                            pdf_stem=pdf_stem,
                            pdf_name=pdf_name,
                            page_text=page_text,
                        )
                    )

                # --- Page crops ---
                if self.crop_pages and self.crop_pages != (1, 1):
                    crop_docs = self._render_page_crops(
                        page=page,
                        out_dir_pages=out_dir_pages,
                        pdf_stem=pdf_stem,
                        pdf_name=pdf_name,
                        page_text=page_text,
                    )
                    page_docs.extend(crop_docs)

                # --- Embedded images (xref) ---
                for info in page.get_image_info(xrefs=True, hashes=False):
                    xref = info.get("xref")
                    smask = info.get("smask")
                    bbox = info.get("bbox")
                    width = int(info.get("width", 0))
                    height = int(info.get("height", 0))

                    if xref is None or min(width, height) < self.min_dim:
                        continue

                    try:
                        data = self._recover_pix(doc, xref, smask)
                        ext = data.get("ext", "png")
                        img_bytes = data["image"]
                        colorspace = data.get("colorspace", 3)

                        if len(img_bytes) < self.min_bytes:
                            continue
                        if self.min_bytes_per_px > 0:
                            if (len(img_bytes) / max(1, (width * height * colorspace))) < self.min_bytes_per_px:
                                continue

                        filename = (
                            f"{pdf_stem}_p{page_num}_x{xref}_w{width}_h{height}.{ext}"
                        )
                        save_path = os.path.join(out_dir_imgs, filename)

                        if self.store_to_folder:
                            with open(save_path, "wb") as f:
                                f.write(img_bytes)

                        image_docs.append(
                            Document(
                                page_content="",
                                metadata={
                                    "type": "image",
                                    "doc_id": pdf_name,
                                    "page": page_num,
                                    "file_path": self.pdf_path,
                                    "image_path": save_path,
                                    "image_ext": ext,
                                    "image_xref": xref,
                                    "width": width,
                                    "height": height,
                                    "bbox": str(bbox),
                                    "has_image": True,
                                    "page_text": page_text,
                                    "url": f"{pdf_name}#page={page_num}",
                                },
                            )
                        )
                    except Exception as ex:
                        # Bad xref / decoding issue: warn and continue.
                        # This ensures pages and crops are still processed.
                        print(
                            f"[WARN] Skipping embedded image xref={xref} on "
                            f"{pdf_name} page {page_num} due to error: {ex}"
                        )
                        continue

        return image_docs, page_docs

    # this is adopted from langchain's DirectoryLoader pattern
    @classmethod
    def load_from_directory(
        cls,
        pdfs_dir,
        images_dir,
        pages_dir=None,
        pages_as_images=False,
        crop_pages: tuple[int, int] | None = None,
        **kwargs,
    ):
        """
        Walk `pdfs_dir` and collect:
        - image Documents to `images_dir`
        - page-image + crop Documents to `pages_dir`
          (if pages_as_images=True or crop_pages is not None)

        Returns (all_image_docs, all_page_docs).
        """
        all_image_docs, all_page_docs = [], []
        pdf_files = [
            os.path.join(pdfs_dir, f)
            for f in os.listdir(pdfs_dir)
            if f.lower().endswith(".pdf")
        ]
        for pdf_path in tqdm(pdf_files, desc="Extracting images/pages from PDFs"):
            loader = cls(
                pdf_path=pdf_path,
                images_root_dir=images_dir,
                pages_as_images=pages_as_images,
                pages_root_dir=pages_dir,
                crop_pages=crop_pages,
                **kwargs,
            )
            try:
                img_docs, page_docs = loader.load()
                all_image_docs.extend(img_docs)
                all_page_docs.extend(page_docs)
            except Exception as ex:
                # Now, only truly fatal errors will skip a whole file. 
                # Individual bad xrefs are handled inside .load().
                print(f"[WARN] Skipping {os.path.basename(pdf_path)} due to error: {ex}")
        return all_image_docs, all_page_docs





def chunk_documents(documents, chunk_size=chunking_size, chunk_step=chunking_step):
    """Splits documents into chunks of 1000 characters with 200 characters step size."""
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_step,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs
    


