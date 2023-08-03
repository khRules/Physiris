import cv2
import math
import os
import pygame
import random
import sys

from ultralytics import YOLO


EVENT_UNICODES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '^', '\\']
EVENT_KEYS = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0, pygame.K_MINUS, pygame.K_CARET, pygame.K_BACKSLASH]
SCORE_MAX = 20
TIME_BEFORE_BEGINNING_GAME_DEFAULT = 2000
TIME_BEFORE_FALL = 1000
TIME_BEFORE_NEXT_CHAR = 1000
TIME_BEFORE_REMOVING_COMPLETE_ROW = 1000
TIME_FADING_COMPLETE_ROW = 266
TIME_GAME_OVER_STAGE_1 = 2000
TIME_GAME_OVER_STAGE_2 = 2000
TIME_GAME_OVER_STAGE_3 = 5000
X_INIT = 6
YOLO_X_MIN_DEFAULT = 0.1
YOLO_X_MAX_DEFAULT = 0.9
YOLO_Y_ORIG_DEFAULT = 0.5
YOLO_Y_ABOVE_DEFAULT = 0.45
YOLO_Y_BELOW_DEFAULT = 0.6


def add_score_for_row_completion():
    global field, score, time_cur, time_last_scored, y_cur

    num_of_rows_removed = 0
    for i in range(y_cur, y_cur + 4):
        for j in range(3, 13):
            if (field[i][j] <= 0) or (field[i][j] >= 13):
                break
        else:
            num_of_rows_removed += 1

    score_prev = score
    score += num_of_rows_removed
    if score != score_prev:
        time_last_scored = time_cur


def complete_row_exists():
    global field, y_cur

    for i in range(y_cur, y_cur + 4):
        for j in range(3, 13):
            if (field[i][j] <= 0) or (field[i][j] >= 13):
                break
        else:
            return True

    return False


def create_character():
    global index_char_cur, index_char_next, rot_cur, rot_new, x_cur, x_new, y_cur, y_new

    if index_char_cur >= 0:
        return False

    index_char_cur = index_char_next
    index_char_next = random.randrange( len(chars) )

    row = chars[index_char_cur][0][0]
    x_cur = X_INIT
    is_first_row_empty = ( (row[0] | row[1] | row[2] | row[3]) == 0 )
    y_cur = 0 if is_first_row_empty else 1
    rot_cur = 0
    x_new, y_new, rot_new = x_cur, y_cur, rot_cur

    return True


def create_surface_of_captured_image():
    global video_cap

    if video_cap is None:
        return None
    ret, frame = video_cap.read()
    if not ret:
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.swapaxes(0, 1)

    return pygame.surfarray.make_surface(frame)


def fade_complete_rows(ratio = 1.0):
    global field, y_cur

    ratio_as_int = int(ratio * 4)
    if ratio_as_int <= 0:
        return

    for i in range(y_cur, y_cur + 4):
        for j in range(3, 13):
            if (field[i][j] <= 0) or (field[i][j] >= 13):
                break
        else:
            index_tile = (9 + ratio_as_int) if ratio_as_int < 4 else 8
            for j in range(3, 13):
                field[i][j] = index_tile


def get_yolo_pos():
    global model, video_cap

    if (model is None) or (video_cap is None):
        return None

    ret, frame = video_cap.read()
    if not ret:
        return None
    height_frame, width_frame = frame.shape[:2]
    mag = 192 / height_frame
    frame = cv2.resize(  frame, dsize=( int(width_frame * mag), 192 )  )
    height_frame_resized, width_frame_resized = frame.shape[:2]

    result = model(frame, imgsz=192)[0]
    x_center_best = 0
    y_center_best = 0
    conf_best = 0
    for data in result.boxes.data:
        xmin, ymin, xmax, ymax, conf, clas = data
        if clas == 0:
            x_center = (xmin + xmax) * 0.5
            y_center = (ymin + ymax) * 0.5
            if conf > conf_best:
                x_center_best = x_center.item()
                y_center_best = y_center.item()
                conf_best = conf
    if conf_best < 0.5:
        return None

    return (x_center_best / width_frame_resized, y_center_best / height_frame_resized)  # 0～1の範囲内に収める。


def gray_field(ratio = 1.0):
    global field

    i_limit = min(  max( int(ratio * 16.0), 0 ), 16  ) + 1
    for i in range(1, i_limit):
        for j in range(3, 13):
            if (field[i][j] > 0) and (field[i][j] < 9):
                field[i][j] = 9


def initialize_characters():
    global chars

    char_1 = [
        [ [0, 0, 0, 0],
          [1, 1, 1, 0],
          [0, 1, 0, 0],
          [0, 0, 0, 0] ],

        [ [0, 1, 0, 0],
          [0, 1, 1, 0],
          [0, 1, 0, 0],
          [0, 0, 0, 0] ],

        [ [0, 0, 0, 0],
          [0, 1, 0, 0],
          [1, 1, 1, 0],
          [0, 0, 0, 0] ],

        [ [0, 1, 0, 0],
          [1, 1, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 0, 0] ]
        ]

    char_2 = [
        [ [0, 0, 0, 0],
          [2, 2, 2, 2],
          [0, 0, 0, 0],
          [0, 0, 0, 0] ],

        [ [0, 0, 2, 0],
          [0, 0, 2, 0],
          [0, 0, 2, 0],
          [0, 0, 2, 0] ],

        [ [0, 0, 0, 0],
          [2, 2, 2, 2],
          [0, 0, 0, 0],
          [0, 0, 0, 0] ],

        [ [0, 0, 2, 0],
          [0, 0, 2, 0],
          [0, 0, 2, 0],
          [0, 0, 2, 0] ]
        ]

    char_3 = [
        [ [0, 0, 0, 0],
          [3, 3, 3, 0],
          [3, 0, 0, 0],
          [0, 0, 0, 0] ],

        [ [0, 3, 0, 0],
          [0, 3, 0, 0],
          [0, 3, 3, 0],
          [0, 0, 0, 0] ],

        [ [0, 0, 0, 0],
          [0, 0, 3, 0],
          [3, 3, 3, 0],
          [0, 0, 0, 0] ],

        [ [3, 3, 0, 0],
          [0, 3, 0, 0],
          [0, 3, 0, 0],
          [0, 0, 0, 0] ]
        ]

    char_4 = [
        [ [0, 0, 0, 0],
          [4, 4, 4, 0],
          [0, 0, 4, 0],
          [0, 0, 0, 0] ],

        [ [0, 4, 4, 0],
          [0, 4, 0, 0],
          [0, 4, 0, 0],
          [0, 0, 0, 0] ],

        [ [0, 0, 0, 0],
          [4, 0, 0, 0],
          [4, 4, 4, 0],
          [0, 0, 0, 0] ],

        [ [0, 4, 0, 0],
          [0, 4, 0, 0],
          [4, 4, 0, 0],
          [0, 0, 0, 0] ]
        ]

    char_5 = [
        [ [0, 5, 5, 0],
          [0, 5, 5, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0] ],

        [ [0, 5, 5, 0],
          [0, 5, 5, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0] ],

        [ [0, 5, 5, 0],
          [0, 5, 5, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0] ],

        [ [0, 5, 5, 0],
          [0, 5, 5, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0] ]
        ]

    char_6 = [
        [ [0, 0, 0, 0],
          [6, 6, 0, 0],
          [0, 6, 6, 0],
          [0, 0, 0, 0] ],

        [ [0, 0, 6, 0],
          [0, 6, 6, 0],
          [0, 6, 0, 0],
          [0, 0, 0, 0] ],

        [ [0, 0, 0, 0],
          [6, 6, 0, 0],
          [0, 6, 6, 0],
          [0, 0, 0, 0] ],

        [ [0, 0, 6, 0],
          [0, 6, 6, 0],
          [0, 6, 0, 0],
          [0, 0, 0, 0] ]
        ]

    char_7 = [
        [ [0, 0, 0, 0],
          [0, 7, 7, 0],
          [7, 7, 0, 0],
          [0, 0, 0, 0] ],

        [ [7, 0, 0, 0],
          [7, 7, 0, 0],
          [0, 7, 0, 0],
          [0, 0, 0, 0] ],

        [ [0, 0, 0, 0],
          [0, 7, 7, 0],
          [7, 7, 0, 0],
          [0, 0, 0, 0] ],

        [ [7, 0, 0, 0],
          [7, 7, 0, 0],
          [0, 7, 0, 0],
          [0, 0, 0, 0] ]
        ]

    chars = [char_1, char_2, char_3, char_4, char_5, char_6, char_7]


def initialize_field():
    # フィールドにおける各要素の値
    # 　0: 何もない
    # 　1～7: キャラクターの構成要素である正方形（可視）
    # 　8: キャラクターの構成要素である正方形（不可視）
    # 　9: キャラクターの構成要素である正方形（可視・区別不可能）
    # 　10～12: 消去中のキャラクターの構成要素である正方形
    # 　13: フィールド周囲の壁（可視）
    # 　14: フィールド周囲の壁（不可視）

    global field

    field = [14] * 2 + [13] + [0] * 10 + [13] + [14] * 2
    field = [field.copy() for i in range(20)]
    field[0] = [14] * 16
    field[-3] = [14] * 2 + [13] * 12 + [14] * 2
    field[-2] = [14] * 16
    field[-1] = [14] * 16


def initialize_surfaces():
    global surface_tiles

    surface_tile_0 = pygame.Surface( (8, 8) )
    surface_tile_1 = pygame.image.load('tile_1.png')
    surface_tile_2 = pygame.image.load('tile_2.png')
    surface_tile_3 = pygame.image.load('tile_3.png')
    surface_tile_4 = pygame.image.load('tile_4.png')
    surface_tile_5 = pygame.image.load('tile_5.png')
    surface_tile_6 = pygame.image.load('tile_6.png')
    surface_tile_7 = pygame.image.load('tile_7.png')
    surface_tile_8 = pygame.Surface( (8, 8) )
    surface_tile_9 = pygame.image.load('tile_9.png')
    surface_tile_10 = pygame.image.load('tile_10.png')
    surface_tile_11 = pygame.image.load('tile_11.png')
    surface_tile_12 = pygame.image.load('tile_12.png')
    surface_tile_13 = pygame.image.load('tile_13.png')
    surface_tile_14 = pygame.Surface( (8, 8) )
    surface_tile_15 = pygame.Surface( (8, 8) )
    surface_tile_16 = pygame.Surface( (8, 8) )
    surface_tile_17 = pygame.Surface( (8, 8) )
    surface_tile_18 = pygame.Surface( (8, 8) )
    surface_tile_19 = pygame.Surface( (8, 8) )
    surface_tile_20 = pygame.Surface( (8, 8) )
    surface_tile_21 = pygame.Surface( (8, 8) )
    surface_tile_22 = pygame.Surface( (8, 8) )
    surface_tile_23 = pygame.Surface( (8, 8) )
    surface_tile_24 = pygame.Surface( (8, 8) )
    surface_tile_25 = pygame.Surface( (8, 8) )
    surface_tile_26 = pygame.Surface( (8, 8) )
    surface_tile_27 = pygame.Surface( (8, 8) )
    surface_tile_28 = pygame.Surface( (8, 8) )
    surface_tile_29 = pygame.Surface( (8, 8) )
    surface_tile_30 = pygame.Surface( (8, 8) )
    surface_tile_31 = pygame.Surface( (8, 8) )
    surface_tile_32 = pygame.Surface( (8, 8) )
    surface_tile_33 = pygame.image.load('char_33.png')
    surface_tile_34 = pygame.image.load('char_34.png')
    surface_tile_35 = pygame.image.load('char_35.png')
    surface_tile_36 = pygame.image.load('char_36.png')
    surface_tile_37 = pygame.image.load('char_37.png')
    surface_tile_38 = pygame.image.load('char_38.png')
    surface_tile_39 = pygame.image.load('char_39.png')
    surface_tile_40 = pygame.image.load('char_40.png')
    surface_tile_41 = pygame.image.load('char_41.png')
    surface_tile_42 = pygame.image.load('char_42.png')
    surface_tile_43 = pygame.Surface( (8, 8) )
    surface_tile_44 = pygame.image.load('char_44.png')
    surface_tile_45 = pygame.image.load('char_45.png')
    surface_tile_46 = pygame.image.load('char_46.png')
    surface_tile_47 = pygame.image.load('char_47.png')
    surface_tile_48 = pygame.image.load('char_48.png')
    surface_tile_49 = pygame.image.load('char_49.png')
    surface_tile_50 = pygame.image.load('char_50.png')
    surface_tile_51 = pygame.image.load('char_51.png')
    surface_tile_52 = pygame.image.load('char_52.png')
    surface_tile_53 = pygame.image.load('char_53.png')
    surface_tile_54 = pygame.image.load('char_54.png')
    surface_tile_55 = pygame.image.load('char_55.png')
    surface_tile_56 = pygame.image.load('char_56.png')
    surface_tile_57 = pygame.image.load('char_57.png')
    surface_tile_58 = pygame.image.load('char_58.png')
    surface_tile_59 = pygame.image.load('char_59.png')
    surface_tile_60 = pygame.image.load('char_60.png')
    surface_tile_61 = pygame.image.load('char_61.png')
    surface_tile_62 = pygame.image.load('char_62.png')
    surface_tile_63 = pygame.image.load('char_63.png')
    surface_tile_64 = pygame.image.load('char_64.png')
    surface_tile_65 = pygame.image.load('char_65.png')
    surface_tile_66 = pygame.image.load('char_66.png')
    surface_tile_67 = pygame.image.load('char_67.png')
    surface_tile_68 = pygame.image.load('char_68.png')
    surface_tile_69 = pygame.image.load('char_69.png')
    surface_tile_70 = pygame.image.load('char_70.png')
    surface_tile_71 = pygame.image.load('char_71.png')
    surface_tile_72 = pygame.image.load('char_72.png')
    surface_tile_73 = pygame.image.load('char_73.png')
    surface_tile_74 = pygame.image.load('char_74.png')
    surface_tile_75 = pygame.image.load('char_75.png')
    surface_tile_76 = pygame.image.load('char_76.png')
    surface_tile_77 = pygame.image.load('char_77.png')
    surface_tile_78 = pygame.image.load('char_78.png')
    surface_tile_79 = pygame.image.load('char_79.png')
    surface_tile_80 = pygame.image.load('char_80.png')
    surface_tile_81 = pygame.image.load('char_81.png')
    surface_tile_82 = pygame.image.load('char_82.png')
    surface_tile_83 = pygame.image.load('char_83.png')
    surface_tile_84 = pygame.image.load('char_84.png')
    surface_tile_85 = pygame.image.load('char_85.png')
    surface_tile_86 = pygame.image.load('char_86.png')
    surface_tile_87 = pygame.image.load('char_87.png')
    surface_tile_88 = pygame.image.load('char_88.png')
    surface_tile_89 = pygame.image.load('char_89.png')
    surface_tile_90 = pygame.image.load('char_90.png')
    surface_tile_91 = pygame.Surface( (8, 8) )
    surface_tile_92 = pygame.Surface( (8, 8) )
    surface_tile_93 = pygame.Surface( (8, 8) )
    surface_tile_94 = pygame.Surface( (8, 8) )
    surface_tile_95 = pygame.Surface( (8, 8) )
    surface_tiles = [surface_tile_0, surface_tile_1,
        surface_tile_2, surface_tile_3,
        surface_tile_4, surface_tile_5,
        surface_tile_6, surface_tile_7,
        surface_tile_8, surface_tile_9,
        surface_tile_10, surface_tile_11,
        surface_tile_12, surface_tile_13,
        surface_tile_14, surface_tile_15,
        surface_tile_16, surface_tile_17,
        surface_tile_18, surface_tile_19,
        surface_tile_20, surface_tile_21,
        surface_tile_22, surface_tile_23,
        surface_tile_24, surface_tile_25,
        surface_tile_26, surface_tile_27,
        surface_tile_28, surface_tile_29,
        surface_tile_30, surface_tile_31,
        surface_tile_32, surface_tile_33,
        surface_tile_34, surface_tile_35,
        surface_tile_36, surface_tile_37,
        surface_tile_38, surface_tile_39,
        surface_tile_40, surface_tile_41,
        surface_tile_42, surface_tile_43,
        surface_tile_44, surface_tile_45,
        surface_tile_46, surface_tile_47,
        surface_tile_48, surface_tile_49,
        surface_tile_50, surface_tile_51,
        surface_tile_52, surface_tile_53,
        surface_tile_54, surface_tile_55,
        surface_tile_56, surface_tile_57,
        surface_tile_58, surface_tile_59,
        surface_tile_60, surface_tile_61,
        surface_tile_62, surface_tile_63,
        surface_tile_64, surface_tile_65,
        surface_tile_66, surface_tile_67,
        surface_tile_68, surface_tile_69,
        surface_tile_70, surface_tile_71,
        surface_tile_72, surface_tile_73,
        surface_tile_74, surface_tile_75,
        surface_tile_76, surface_tile_77,
        surface_tile_78, surface_tile_79,
        surface_tile_80, surface_tile_81,
        surface_tile_82, surface_tile_83,
        surface_tile_84, surface_tile_85,
        surface_tile_86, surface_tile_87,
        surface_tile_88, surface_tile_89,
        surface_tile_90, surface_tile_91,
        surface_tile_92, surface_tile_93,
        surface_tile_94, surface_tile_95]


def insert_new_character(forces = False):
    global chars, field, index_char_cur, rot_new, x_new, y_new

    if index_char_cur < 0:
        return False
    char_cur = chars[index_char_cur][rot_new]

    for i in range( len(char_cur) ):
        for j in range( len(char_cur[i]) ):
            if (char_cur[i][j] != 0) and (field[y_new + i][x_new + j] != 0) and not forces:
                return False

    for i in range( len(char_cur) ):
        for j in range( len(char_cur[i]) ):
            if char_cur[i][j] != 0:
                field[y_new + i][x_new + j] = char_cur[i][j]

    return True


def remove_complete_rows():
    global field, score, time_cur, time_last_scored, y_cur

    for i in range(y_cur, y_cur + 4):
        for j in range(3, 13):
            if (field[i][j] <= 0) or (field[i][j] >= 13):
                break
        else:
            for i2 in range(i, 1, -1):
                for j in range(3, 13):
                    field[i2][j] = field[i2 - 1][j]  # 1行分下にずらす。
            for j in range(3, 13):
                field[1][j] = 0  # 最上行を「何もない」にする。


def remove_current_character():
    global chars, field, index_char_cur, rot_cur, x_cur, y_cur

    if index_char_cur < 0:
        return
    char_cur = chars[index_char_cur][rot_cur]

    for i in range( len(char_cur) ):
        for j in range( len(char_cur[i]) ):
            if char_cur[i][j] != 0:
                field[y_cur + i][x_cur + j] = 0


def set_yolo_x_max():
    global yolo_x_max

    yolo_pos = get_yolo_pos()
    if yolo_pos is not None:
        yolo_x_max = yolo_pos[0]


def set_yolo_x_min():
    global yolo_x_min

    yolo_pos = get_yolo_pos()
    if yolo_pos is not None:
        yolo_x_min = yolo_pos[0]


def set_yolo_y_above():
    global yolo_y_above

    yolo_pos = get_yolo_pos()
    if yolo_pos is not None:
        yolo_y_above = yolo_pos[1]


def set_yolo_y_below():
    global yolo_y_below

    yolo_pos = get_yolo_pos()
    if yolo_pos is not None:
        yolo_y_below = yolo_pos[1]


def set_yolo_y_orig():
    global yolo_y_orig

    yolo_pos = get_yolo_pos()
    if yolo_pos is not None:
        yolo_y_orig = yolo_pos[1]


def show_score(surface, score, x_orig = 208, y_orig = 48):
    global surface_tiles

    surface.blit( surface_tiles[83], (x_orig - 7 * 8, y_orig - 8) )
    surface.blit( surface_tiles[67], (x_orig - 6 * 8, y_orig - 8) )
    surface.blit( surface_tiles[79], (x_orig - 5 * 8, y_orig - 8) )
    surface.blit( surface_tiles[82], (x_orig - 4 * 8, y_orig - 8) )
    surface.blit( surface_tiles[69], (x_orig - 3 * 8, y_orig - 8) )

    tmp_score = score
    for i in range(8):
        if (i > 0) and (tmp_score <= 0):
            break
        digit = tmp_score % 10
        tmp_score //= 10
        surface.blit( surface_tiles[digit + 48], (x_orig - i * 8, y_orig) )


def show_time_elapsed(surface, time_elapsed, x_orig = 208, y_orig = 72, caption='TIME'):
    global surface_tiles

    for i, c in enumerate(caption):
        i_c = ord(c)
        if i_c < len(surface_tiles):
            surface.blit(  surface_tiles[i_c], ( x_orig + (i - 7) * 8, y_orig - 8 )  )

    time_elapsed_centisec = (time_elapsed % 1000) // 10
    time_elapsed //= 1000
    for i in range(2):
        digit = time_elapsed_centisec % 10
        time_elapsed_centisec //= 10
        surface.blit( surface_tiles[digit + 48], (x_orig - i * 8, y_orig) )
    surface.blit( surface_tiles[46], (x_orig - 16, y_orig) )
    time_elapsed_sec = time_elapsed % 60
    time_elapsed //= 60
    for i in range(2):
        if (i > 0) and (time_elapsed_sec <= 0) and (time_elapsed <= 0):
            break
        digit = time_elapsed_sec % 10
        time_elapsed_sec //= 10
        surface.blit( surface_tiles[digit + 48], (x_orig - 24 - i * 8, y_orig) )
    if time_elapsed > 0:
        surface.blit( surface_tiles[58], (x_orig - 40, y_orig) )
    time_elapsed_minute = time_elapsed
    for i in range(2):
        if time_elapsed_minute <= 0:
            break
        digit = time_elapsed_minute % 10
        time_elapsed_minute //= 10
        surface.blit( surface_tiles[digit + 48], (x_orig - 48 - i * 8, y_orig) )


def simulate_keydown_by_yolo(reverses_x=False):
    global time_cur, x_cur, yolo_jumps, yolo_time_of_last_squat, yolo_x_max, yolo_x_min, yolo_y_above, yolo_y_below, yolo_y_orig

    yolo_pos = get_yolo_pos()
    if yolo_pos is None:
        return
    yolo_x, yolo_y = yolo_pos

    yolo_y_above_range = yolo_y_orig - yolo_y_above
    if yolo_y_above_range >= 0.001:
        threshold = 0.5 if yolo_jumps else 0.7
        yolo_jumps = (yolo_y_orig - yolo_y) / yolo_y_above_range >= threshold
        if yolo_jumps:
            yolo_jumps = True
            newevent = pygame.event.Event(pygame.KEYDOWN, unicode=' ', key=pygame.K_SPACE, mod=pygame.KMOD_NONE)
            pygame.event.post(newevent)
            return

    yolo_y_below_range = yolo_y_below - yolo_y_orig
    if yolo_y_below_range >= 0.001:
        threshold = 0.5 if yolo_time_of_last_squat > 0 else 0.7
        if (yolo_y - yolo_y_orig) / yolo_y_below_range >= threshold:
            if (yolo_time_of_last_squat < 0) or (time_cur - yolo_time_of_last_squat >= 250):
                yolo_time_of_last_squat = time_cur
                newevent = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, mod=pygame.KMOD_NONE)
                pygame.event.post(newevent)
                return
        else:
            yolo_time_of_last_squat = -1

    yolo_x_range = yolo_x_max - yolo_x_min
    if yolo_x_range >= 0.001:
        x_tmp = (yolo_x - yolo_x_min) / yolo_x_range * 13
        if not reverses_x:
            x_tmp = 13 - x_tmp
        if abs( x_tmp - (x_cur + 0.5) ) >= 0.75:
            x_tmp = min(   max(  int( math.floor(x_tmp) ), 0  ), 12   )
            newevent = pygame.event.Event(pygame.KEYDOWN, unicode=EVENT_UNICODES[x_tmp], key=EVENT_KEYS[x_tmp], mod=pygame.KMOD_NONE)
            pygame.event.post(newevent)
            return


initialize_field()
initialize_characters()
initialize_surfaces()

pygame.init()
pygame.display.set_caption('Physiris')
# screen = pygame.display.set_mode( (640, 400) )
screen = pygame.display.set_mode( (1280, 800) )
black = (0, 0, 0)
white = (255, 255, 255)

is_music_loaded = False

time_before_beginning_game = TIME_BEFORE_BEGINNING_GAME_DEFAULT

game_mode = 0  # 0: なし（ゲームが開始されていない）　# 1～2: ゲーム開始準備　3: ゲーム操作受付中　4: 行消去　5～8: ゲームオーバー演出
time_start = 0
time_end = 0
time_cur = -1
time_prev = -1
is_game_successful = False

score = 0
time_last_scored = 0

index_char_cur = -1
index_char_next = -1
x_cur = X_INIT
y_cur = 1
rot_cur = 0

x_new, y_new, rot_new = x_cur, y_cur, rot_cur

time_remain = -1

moves_horiz = 0
moves_vert = 0

model = YOLO('yolov8n.pt')
video_cap = cv2.VideoCapture(0)
# video_cap = cv2.VideoCapture('20230707_2259.mp4')
enables_yolo = False
shows_captured_img = False
yolo_x_min = YOLO_X_MIN_DEFAULT
yolo_x_max = YOLO_X_MAX_DEFAULT
yolo_y_orig = YOLO_Y_ORIG_DEFAULT
yolo_y_above = YOLO_Y_ABOVE_DEFAULT
yolo_y_below = YOLO_Y_BELOW_DEFAULT
yolo_jumps = False
yolo_time_of_last_squat = -1

quits = False
surface_primary = pygame.Surface( (320, 200) )
surface_captured_img = None
screen.fill(black)
time_prev = pygame.time.get_ticks()
while True:
    time_cur = pygame.time.get_ticks()
    # print(f'score: {score}  index_char_cur: {index_char_cur}  game_mode: {game_mode}  time_remain: {time_remain}  x_cur: {x_cur}  y_cur: {y_cur}')
    time_diff = time_cur - time_prev

    if enables_yolo:
        simulate_keydown_by_yolo()

    if moves_horiz == 0:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    moves_horiz = 9 - x_cur
                elif event.key == pygame.K_1:
                    moves_horiz = -x_cur
                elif event.key == pygame.K_2:
                    moves_horiz = 1 - x_cur
                elif event.key == pygame.K_3:
                    moves_horiz = 2 - x_cur
                elif event.key == pygame.K_4:
                    moves_horiz = 3 - x_cur
                elif event.key == pygame.K_5:
                    moves_horiz = 4 - x_cur
                elif event.key == pygame.K_6:
                    moves_horiz = 5 - x_cur
                elif event.key == pygame.K_7:
                    moves_horiz = 6 - x_cur
                elif event.key == pygame.K_8:
                    moves_horiz = 7 - x_cur
                elif event.key == pygame.K_9:
                    moves_horiz = 8 - x_cur
                elif event.key == pygame.K_BACKSLASH:
                    moves_horiz = 12 - x_cur
                elif event.key == pygame.K_CARET:
                    moves_horiz = 11 - x_cur
                elif event.key == pygame.K_DOWN:
                    moves_vert = 1
                    if index_char_cur >= 0:
                        time_remain = TIME_BEFORE_FALL + time_diff
                elif event.key == pygame.K_ESCAPE:
                    quits = True
                    break
                elif event.key == pygame.K_LEFT:
                    moves_horiz = -1
                elif event.key == pygame.K_MINUS:
                    moves_horiz = 10 - x_cur
                elif event.key == pygame.K_RETURN:
                    if game_mode == 0:
                        game_mode = 1
                elif event.key == pygame.K_RIGHT:
                    moves_horiz = 1
                elif event.key == pygame.K_SPACE:
                    rot_new = (rot_cur + 1) & 3
                elif event.key == pygame.K_a:
                    yolo_x_min = YOLO_X_MIN_DEFAULT
                elif event.key == pygame.K_b:
                    set_yolo_y_below()
                elif event.key == pygame.K_c:
                    set_yolo_y_orig()
                elif event.key == pygame.K_d:
                    yolo_y_orig = YOLO_Y_ORIG_DEFAULT
                elif event.key == pygame.K_f:
                    yolo_y_above = YOLO_Y_ABOVE_DEFAULT
                elif event.key == pygame.K_g:
                    yolo_y_below = YOLO_Y_BELOW_DEFAULT
                elif event.key == pygame.K_o:
                    shows_captured_img = True
                elif event.key == pygame.K_p:
                    shows_captured_img = False
                elif event.key == pygame.K_s:
                    yolo_x_max = YOLO_X_MAX_DEFAULT
                elif event.key == pygame.K_u:
                    enables_yolo = False
                elif event.key == pygame.K_v:
                    set_yolo_y_above()
                elif event.key == pygame.K_x:
                    set_yolo_x_max()
                elif event.key == pygame.K_y:
                    enables_yolo = True
                elif event.key == pygame.K_z:
                    set_yolo_x_min()
            if event.type == pygame.QUIT:
                quits = True
                break

    if quits:
        break

    if game_mode != 0:
        time_remain -= time_diff

    if (game_mode == 3) and (time_remain < 0) and (index_char_cur >= 0):
        moves_vert = 1

    if moves_vert > 0:
        y_new += 1
        moves_vert -= 1
    elif moves_horiz < 0:
        x_new -= 1
        moves_horiz += 1
    elif moves_horiz > 0:
        x_new += 1
        moves_horiz -= 1

    inserting_char_successful = False
    is_next_char_just_created = False

    if game_mode == 1:
        game_mode = 2
        initialize_field()
        is_game_successful = False
        score = 0
        time_end = time_start
        time_last_scored = time_start
        index_char_cur = -1
        time_remain = time_before_beginning_game
        if os.path.isfile('bgm_2.wav'):
            pygame.mixer.music.load('bgm_2.wav')
            is_music_loaded = True
    elif game_mode == 2:
        if time_remain < 0:
            game_mode = 3
            time_start = pygame.time.get_ticks()
            time_end = -1
            time_last_scored = time_start
            index_char_next = random.randrange( len(chars) )
            time_remain = TIME_BEFORE_NEXT_CHAR
            if is_music_loaded:
                pygame.mixer.music.play(-1)
    elif game_mode == 3:
        if time_remain < 0:
            if index_char_cur < 0:
                is_next_char_just_created = create_character()
            time_remain += TIME_BEFORE_FALL

        inserting_char_successful = insert_new_character()
        if inserting_char_successful:
            x_cur, y_cur, rot_cur = x_new, y_new, rot_new
        else:
            if is_next_char_just_created:  # キャラクターが上まで積みあがった場合。
                time_end = pygame.time.get_ticks()
                game_mode = 5  # ゲーム終了。
                insert_new_character(True)
            elif index_char_cur < 0:  # 現在のキャラクターがない場合。
                pass
            elif y_cur != y_new:  # キャラクターがこれ以上下には行けない場合。
                x_new, y_new, rot_new = x_cur, y_cur, rot_cur
                inserting_char_successful = insert_new_character()
                if complete_row_exists():
                    add_score_for_row_completion()
                    if score >= SCORE_MAX:
                        time_end = pygame.time.get_ticks()
                        is_game_successful = True
                    game_mode = 4
                    time_remain += TIME_BEFORE_REMOVING_COMPLETE_ROW - TIME_BEFORE_FALL
                else:
                    time_remain += TIME_BEFORE_NEXT_CHAR - TIME_BEFORE_FALL
                index_char_cur = -1
            else:
                x_new, y_new, rot_new = x_cur, y_cur, rot_cur
                inserting_char_successful = insert_new_character()
    elif game_mode == 4:
        fade_complete_rows( (TIME_BEFORE_REMOVING_COMPLETE_ROW - time_remain) / TIME_FADING_COMPLETE_ROW )
        if time_remain < 0:
            if score >= SCORE_MAX:
                game_mode = 5  # ゲーム終了。
            else:
                game_mode = 3
                remove_complete_rows()
                time_remain += TIME_BEFORE_NEXT_CHAR - TIME_BEFORE_FALL
    elif game_mode == 5:
        game_mode = 6
        time_remain += TIME_GAME_OVER_STAGE_1 - TIME_BEFORE_FALL
        if is_music_loaded:
            pygame.mixer.music.stop()
    elif game_mode == 6:
        if time_remain < 0:
            game_mode = 7
            if not is_game_successful:
                gray_field()
            time_remain += TIME_GAME_OVER_STAGE_2
        else:
            if not is_game_successful:
                gray_field(1.0 - time_remain / TIME_GAME_OVER_STAGE_1)
    elif game_mode == 7:
        if time_remain < 0:
            game_mode = 8
            time_remain += TIME_GAME_OVER_STAGE_3
    elif game_mode == 8:
        if time_remain < 0:
            game_mode = 0

    surface_primary.fill(black)

    if game_mode != 2:
        char_next = chars[index_char_next][0]
        row = char_next[2]
        is_row_empty = ( (row[0] | row[1] | row[2] | row[3]) == 0 )
        for y in range( len(char_next) ):
            for x in range( len(char_next[y]) ):
                surface_primary.blit(  surface_tiles[ char_next[y][x] ], ( (x + X_INIT) * 8 + 16, y * 8 + (16 if is_row_empty else 8) )  )

        surface_primary.blit(  surface_tiles[78], ( (X_INIT - 4) * 8 + 16, 24 )  )
        surface_primary.blit(  surface_tiles[69], ( (X_INIT - 3) * 8 + 16, 24 )  )
        surface_primary.blit(  surface_tiles[88], ( (X_INIT - 2) * 8 + 16, 24 )  )
        surface_primary.blit(  surface_tiles[84], ( (X_INIT - 1) * 8 + 16, 24 )  )

    for y in range( len(field) ):
        for x in range( len(field[y]) ):
            surface_primary.blit( surface_tiles[ field[y][x] ], (x * 8 + 16, y * 8 + 32) )

    if game_mode == 8:
        surface_primary.blit( surface_tiles[71], (40, 96) )
        surface_primary.blit( surface_tiles[65], (48, 96) )
        surface_primary.blit( surface_tiles[77], (56, 96) )
        surface_primary.blit( surface_tiles[69], (64, 96) )
        surface_primary.blit( surface_tiles[79], (80, 96) )
        surface_primary.blit( surface_tiles[86], (88, 96) )
        surface_primary.blit( surface_tiles[69], (96, 96) )
        surface_primary.blit( surface_tiles[82], (104, 96) )

    show_score(surface_primary, score)

    show_time_elapsed(surface_primary, time_last_scored - time_start, 208, 64, caption='')
    show_time_elapsed( surface_primary, (time_cur if time_end < 0 else time_end) - time_start, 208, 88 )

    width_screen, height_screen = pygame.display.get_surface().get_size()
    width_surface, height_surface = surface_primary.get_size()
    mag = min(width_screen / width_surface, height_screen / height_surface)
    # print(f'{width_screen} {width_surface} {height_screen} {height_surface}  {mag}')
    surface_mag = pygame.transform.scale( surface_primary, (width_surface * mag, height_surface * mag) )
    x_surface = (width_screen - width_surface * mag) * 0.5
    y_surface = (height_screen - height_surface * mag) * 0.5
    screen.blit( surface_mag, (x_surface, y_surface) )

    # print(f'*** 1  {field}')

    if inserting_char_successful:
        remove_current_character()

    # print(f'*** 2  {field}')

    if shows_captured_img:
        surface_captured_img = create_surface_of_captured_image()
        if surface_captured_img is not None:
            width_surface, height_surface = surface_captured_img.get_size()
            mag = min(width_screen / width_surface, height_screen / height_surface)
            surface_captured_img_mag = pygame.transform.scale( surface_captured_img, (width_surface * mag, height_surface * mag) )
            x_surface = (width_screen - width_surface * mag) * 0.5
            y_surface = (height_screen - height_surface * mag) * 0.5
            screen.blit( surface_captured_img_mag, (x_surface, y_surface) )
    else:
        surface_captured_img = None

    pygame.display.update()

    time_prev = time_cur

if is_music_loaded:
    pygame.mixer.music.stop()
pygame.quit()
sys.exit()
