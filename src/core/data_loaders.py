from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from tqdm import tqdm
from .config import documents_path, chunking_size, chunking_step

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
    


