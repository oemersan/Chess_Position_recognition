from board_detector import BoardDetector
from orientation_detector import detect_orientation
from piece_detector import PieceDetectorYOLOV11, PieceDetectorRTDETR
from fen_generator import FenGenerator

import sys
import cv2
from PIL import Image
import numpy as np
from typing import Union
import chess
import chess.engine
from dataclasses import dataclass
from typing import Optional, List

class ChessPositionDetector:
    def __init__(self, pose_model_path="models/medium_best.pt", resnet_model_path="models/corner_detector_v2_test1.pth", use_resnet=False, piece_model_path_yolo="models/piece_s50.pt", piece_model_path_detr="models/rtdetr.pt", use_detr=True, show=False):
        self.show = show
        self.use_resnet = use_resnet
        if use_resnet:
            self.board_corner_detector = BoardDetector(resnet_model_path, use_resnet=True)
        else:
            self.board_corner_detector = BoardDetector(pose_model_path, use_resnet=False)
        if use_detr:
            self.piece_detector = PieceDetectorRTDETR(piece_model_path_detr)
        else:
            self.piece_detector = PieceDetectorYOLOV11(piece_model_path_yolo)
        self.fen_generator = FenGenerator()

    def __call__(self, image: Union[Image.Image, np.ndarray], debug=False):
        pil_image: Image.Image
        cv2_image: np.ndarray

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
            rgb = np.array(pil_image)
            cv2_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        elif isinstance(image, np.ndarray):
            cv2_image = image
            if cv2_image.ndim == 2:
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
            elif cv2_image.ndim == 3 and cv2_image.shape[2] == 4:
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGRA2BGR)
            elif cv2_image.ndim == 3 and cv2_image.shape[2] == 3:
                pass
            else:
                raise ValueError(f"Unsupported ndarray shape: {cv2_image.shape}")

            pil_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
        if self.show:
            cv2.imshow("original_image", image)
            cv2.waitKey(1)

        corners, warped_board, H, wide_warped_board = self.board_corner_detector(cv2_image)
        if corners is None:
            return None
        
        if self.show:
            cv2.imshow("warped_board", warped_board)
            cv2.waitKey(1)

        if isinstance(wide_warped_board, np.ndarray):
            wide_warped_board = Image.fromarray(cv2.cvtColor(wide_warped_board, cv2.COLOR_BGR2RGB))
        white_edge = detect_orientation(wide_warped_board, debug=debug)
        if white_edge is None:
            for bw_threshold in [110, 130, 90, 150, 70]:
                white_edge = detect_orientation(wide_warped_board, use_inv=True, bw_threshold=bw_threshold, debug=debug)
                if white_edge is not None:
                    if debug:
                        print(f"Detected white edge with bw_threshold={bw_threshold}")
                    break

        if debug:
            print("White edge", white_edge)
        if white_edge is None:
            print("!!!! Failed to detect board orientation.")
            white_edge = 0

        pieces = self.piece_detector(cv2_image)
        if pieces is None:
            return None

        cpy_image = cv2_image.copy()
        for piece in pieces:
            x1, y1, x2, y2, piece_class, conf = piece
            cv2.rectangle(cpy_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        for piece in pieces:
            x1, y1, x2, y2, piece_class, conf = piece
            cv2.putText(cpy_image, self.fen_generator.piece_classes[piece_class], (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if self.show:
            cv2.imshow("detected_pieces", cpy_image)
            cv2.waitKey(1)


        fen = self.fen_generator(pieces, warped_board, H, white_edge, debug=False, original_image=cv2_image)
        if fen is None:
            return None

        return fen


@dataclass
class EngineResult:
    best_move_uci: str
    best_move_san: str
    score_cp: Optional[int]
    mate_in: Optional[int]
    pv_uci: List[str]
    depth: Optional[int]

class ChessEngine:
    def __init__(self, stockfish_path: str = "/usr/games/stockfish",
                 threads: int = 4, hash_mb: int = 256):
        self.stockfish_path = stockfish_path
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def start(self):
        if self._engine is not None:
            return
        self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        try:
            self._engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        except Exception:
            pass

    def close(self):
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @staticmethod
    def _score_to_parts(score: chess.engine.PovScore):
        """
        PovScore -> (cp, mate)
        """
        s = score.white()
        if s.is_mate():
            return None, s.mate()
        return s.score(mate_score=100000), None

    def best_move(self, fen: str, time_limit: float = 0.10, depth: Optional[int] = None) -> EngineResult:
        self.start()
        assert self._engine is not None

        board = chess.Board(fen)

        limit = chess.engine.Limit(time=time_limit) if depth is None else chess.engine.Limit(depth=depth)
        play_res = self._engine.play(board, limit)

        best_move = play_res.move
        best_move_uci = best_move.uci()
        best_move_san = board.san(best_move)

        info = self._engine.analyse(board, limit)
        score = info.get("score")
        depth_out = info.get("depth")
        pv = info.get("pv", [])

        score_cp, mate_in = (None, None)
        if score is not None:
            score_cp, mate_in = self._score_to_parts(score)

        pv_uci = [m.uci() for m in pv]

        return EngineResult(
            best_move_uci=best_move_uci,
            best_move_san=best_move_san,
            score_cp=score_cp,
            mate_in=mate_in,
            pv_uci=pv_uci,
            depth=depth_out
        )

    def analyse(self, fen: str, depth: int = 15) -> EngineResult:
        self.start()
        assert self._engine is not None

        board = chess.Board(fen)
        info = self._engine.analyse(board, chess.engine.Limit(depth=depth))

        pv = info.get("pv", [])
        score = info.get("score")
        depth_out = info.get("depth", depth)

        if pv:
            first = pv[0]
        else:
            first = self._engine.play(board, chess.engine.Limit(depth=depth)).move

        best_move_uci = first.uci()
        best_move_san = board.san(first)

        score_cp, mate_in = (None, None)
        if score is not None:
            score_cp, mate_in = self._score_to_parts(score)

        pv_uci = [m.uci() for m in pv]

        return EngineResult(
            best_move_uci=best_move_uci,
            best_move_san=best_move_san,
            score_cp=score_cp,
            mate_in=mate_in,
            pv_uci=pv_uci,
            depth=depth_out
        )




if __name__ == "__main__":
    import os

    images_path = "images"
    images_path = "/home/mkb/Desktop/DL_PROJECT/corner_detection/test_images"
    images = os.listdir(images_path)
    images = [f for f in images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    detector = ChessPositionDetector(use_resnet=False, show=True, use_detr=True)
    engine = ChessEngine(stockfish_path="/usr/games/stockfish", threads=4, hash_mb=256)

    for image_path in images:
        print(f"Processing image: {image_path}")
        # image = Image.open(os.path.join(images_path, image_path))
        image = cv2.imread(os.path.join(images_path, image_path))
        fen_base = detector(image, debug=False)
        print(fen_base)
        print("---")
        try:
            fen_white = fen_base + " w - - 0 1"
            with engine:
                res = engine.best_move(fen_white, time_limit=0.2)
                print("Best (SAN) for white:", res.best_move_san)
                # print("Best (UCI) for white:", res.best_move_uci)
                print("Score cp for white:", res.score_cp, "Mate in:", res.mate_in)
                # print("PV:", res.pv_uci[:10])
            print("---")
        except Exception as e:
            print("Engine error for white:", e)
        try:
            fen_black = fen_base + " b - - 0 1"
            with engine:
                res = engine.best_move(fen_black, time_limit=0.2)
                print("Best (SAN) for black:", res.best_move_san)
                # print("Best (UCI) for black:", res.best_move_uci)
                print("Score cp for black:", res.score_cp, "Mate in:", res.mate_in)
        except Exception as e:
            print("Engine error for black:", e)
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)
        print("\n"*2, "#"*50, "\n"*2)