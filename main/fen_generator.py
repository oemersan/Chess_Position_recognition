from shapely.geometry import Polygon
import numpy as np
import sys

class FenGenerator:
    def __init__(self):
        self.piece_classes = [
            "b",
            "k",
            "n",
            "p",
            "q",
            "r",
            "B",
            "K",
            "N",
            "P",
            "Q",
            "R"
        ]

        letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
        self.board_labels = []
        for i in range(8):
            row = []
            for j in range(8):
                row.append(f"{letters[j]}{i+1}")
            self.board_labels.append(row)

    def get_square_coords(self, warped_board, white_edge):
        image_size, _ = warped_board.shape[0], warped_board.shape[1]
        square_size = image_size // 8
        board = [[None for _ in range(8)] for _ in range(8)]
        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size

                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2

                if white_edge == 0:
                    board_row = 8-row-1
                    board_col = col
                elif white_edge == 1:
                    board_row = col
                    board_col = row
                elif white_edge == 2:
                    board_row = row
                    board_col = 8-col-1
                elif white_edge == 3:
                    board_row = 8-col-1
                    board_col = 8-row-1
                board[board_row][board_col] = (x_center, y_center)

        return board

    def iou_rect_trapezoid(self, rect_xyxy, trap_points):
        x1, y1, x2, y2 = rect_xyxy
        rect_poly = Polygon([(x1,y1), (x2,y1), (x2,y2), (x1,y2)])
        trap_poly = Polygon(trap_points)

        rect_poly = rect_poly.buffer(0)
        trap_poly = trap_poly.buffer(0)

        inter = rect_poly.intersection(trap_poly).area
        union = rect_poly.union(trap_poly).area
        return 0.0 if union == 0 else inter / union

    def warp_point(self, H, point, image_width, image_height):
        p = np.array([point[0], point[1], 1]).reshape(3, 1)
        wp = np.dot(H, p)
        wp /= wp[2, 0]
        return (min(max(0, int(wp[0, 0])), image_width-1), min(max(0, int(wp[1, 0])), image_height-1))
    
    def fen_from_board(self, board):
        fen_rows = []
        for row in board[::-1]:
            fen_row = ""
            empty_count = 0
            for piece in row:
                if piece == -1:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += self.piece_classes[piece]
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        fen = "/".join(fen_rows)
        return fen
    
    def calculate_best_square(self, warped_board, square_coords, piece_trap, piece_class, conf, debug=False, cpy_warped_board=None):
        max_iou = 0
        best_square = None
        best_square_idx = None
        for row in range(8):
            for col in range(8):
                square_center = square_coords[row][col]
                square_size = warped_board.shape[0] // 8
                half_size = square_size // 2
                square_rect = (
                    square_center[0] - half_size,
                    square_center[1] - half_size,
                    square_center[0] + half_size,
                    square_center[1] + half_size
                )

                iou = self.iou_rect_trapezoid(square_rect, piece_trap)
                if iou > max_iou:
                    max_iou = iou
                    best_square = square_rect
                    best_square_idx = (row, col)
                if debug:
                    import cv2
                    if iou == 0:
                        continue
                    board_label = self.board_labels[row][col]
                    print(f"Piece {self.piece_classes[piece_class]} vs square ({row},{col}) IoU: {iou:.2f},  Conf: {conf:.2f}")
                    cv2.putText(cpy_warped_board, board_label, (square_rect[0] + 20, square_rect[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.rectangle(cpy_warped_board, (square_rect[0], square_rect[1]), (square_rect[2], square_rect[3]), (255, 0, 0), 2)

        return best_square, best_square_idx


    def __call__(self, pieces, warped_board, H, white_edge, debug=False, original_image=None):
        img_height, img_width = warped_board.shape[0], warped_board.shape[1]
        square_coords = self.get_square_coords(warped_board, white_edge)

        board = [[-1 for _ in range(8)] for _ in range(8)]
        for piece in pieces:
            x1, y1, x2, y2, piece_class, conf = piece
            w = x2 - x1
            h = y2 - y1
            cropped_piece_coordinates = [
                (x1+w*0.1, y1+h*0.5),
                (x2-w*0.1, y1+h*0.5),
                (x1+w*0.1, y2-h*0.1),
                (x2-w*0.1, y2-h*0.1)
            ]
            piece_trap_croped = [self.warp_point(H, coord, img_width, img_height) for coord in cropped_piece_coordinates]
            piece_trap_original = [
                self.warp_point(H, (x1, y1), img_width, img_height),
                self.warp_point(H, (x2, y1), img_width, img_height),
                self.warp_point(H, (x1, y2), img_width, img_height),
                self.warp_point(H, (x2, y2), img_width, img_height)
            ]

            if debug:
                import cv2
                cpy_warped_board = warped_board.copy()
                # for coord in piece_trap_croped:
                #     cv2.circle(cpy_warped_board, coord, 5, (0, 0, 255), -1)
                cv2.line(cpy_warped_board, piece_trap_croped[0], piece_trap_croped[1], (0, 0, 255), 2)
                cv2.line(cpy_warped_board, piece_trap_croped[1], piece_trap_croped[3], (0, 0, 255), 2)
                cv2.line(cpy_warped_board, piece_trap_croped[3], piece_trap_croped[2], (0, 0, 255), 2)
                cv2.line(cpy_warped_board, piece_trap_croped[2], piece_trap_croped[0], (0, 0, 255), 2)

            best_square, best_square_idx = self.calculate_best_square(
                warped_board,
                square_coords,
                piece_trap_croped,
                piece_class,
                conf,
                debug=debug,
                cpy_warped_board=cpy_warped_board if debug else None
            )

            if best_square is None:
                best_square, best_square_idx = self.calculate_best_square(
                    warped_board,
                    square_coords,
                    piece_trap_original,
                    piece_class,
                    conf,
                    debug=debug,
                    cpy_warped_board=cpy_warped_board if debug else None
                )

            if best_square is None:
                print(f"Warning: No square found for piece {self.piece_classes[piece_class]} at {x1},{y1},{x2},{y2}")
                continue

            if debug:
                print(f"Best square for piece {self.piece_classes[piece_class]} is {best_square_idx}, {self.board_labels[best_square_idx[0]][best_square_idx[1]]}")
                if original_image is not None:
                    cpy_org_img = original_image.copy()
                    cv2.rectangle(cpy_org_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(cpy_org_img, self.piece_classes[piece_class], (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imshow("Piece in original image", cpy_org_img)
                cv2.rectangle(cpy_warped_board, (best_square[0], best_square[1]), (best_square[2], best_square[3]), (0, 255, 0), 2)
                cv2.imshow("Piece result on warped board", cpy_warped_board)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    sys.exit(0)
            if best_square is not None:
                board[best_square_idx[0]][best_square_idx[1]] = piece_class

        fen = self.fen_from_board(board)
        return fen