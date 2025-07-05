# srcfolder/hud_renderer.py

import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Dict, Any, List, Tuple
from datetime import datetime
import numpy as np
import cv2 # For Canny and HoughLinesP

# Import constants for UI styling and fonts
from .constants import (
    HUD_BLUE_DARK_TRANSPARENT, HUD_BLUE_MEDIUM_TRANSPARENT, HUD_BLUE_LIGHT,
    HUD_CYAN_LIGHT, HUD_GREEN_LIGHT, HUD_YELLOW_ACCENT, HUD_RED_CRITICAL,
    HUD_TEXT_COLOR_PRIMARY, HUD_TEXT_COLOR_SECONDARY, HUD_TEXT_COLOR_HIGHLIGHT,
    HUD_OUTLINE_WIDTH_BASE, HUD_CORNER_RADIUS_BASE, HUD_PADDING_BASE,
    TEXT_OUTLINE_COLOR, TEXT_OUTLINE_WIDTH,
    UI_DESIGN_BASE_WIDTH, UI_DESIGN_BASE_HEIGHT,
    PANEL_BASE_WIDTH_RATIO, PANEL_BASE_HEIGHT_RATIO,
    SCENE_LABEL_COLORS, LANE_LINE_COLOR, LANE_LINE_THICKNESS,
    DEFAULT_FONT_PATH
)

class HUDRenderer:
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.global_ui_scale_w = self.frame_width / UI_DESIGN_BASE_WIDTH
        self.global_ui_scale_h = self.frame_height / UI_DESIGN_BASE_HEIGHT
        self.global_ui_scale = min(self.global_ui_scale_w, self.global_ui_scale_h)

        self.hud_font_size_main_scaled = max(16, int(30 * self.global_ui_scale))
        self.hud_font_size_sub_scaled = max(12, int(22 * self.global_ui_scale))
        self.hud_font_size_small_scaled = max(10, int(16 * self.global_ui_scale))

        try:
            if os.path.exists(DEFAULT_FONT_PATH):
                self.font_main = ImageFont.truetype(DEFAULT_FONT_PATH, self.hud_font_size_main_scaled)
                self.font_sub = ImageFont.truetype(DEFAULT_FONT_PATH, self.hud_font_size_sub_scaled)
                self.font_small = ImageFont.truetype(DEFAULT_FONT_PATH, self.hud_font_size_small_scaled)
                print(f"[INFO] Custom font loaded from: {DEFAULT_FONT_PATH}")
            else:
                self.font_main = ImageFont.load_default()
                self.font_sub = ImageFont.load_default()
                self.font_small = ImageFont.load_default()
                print(f"[WARNING] Custom font not found at {DEFAULT_FONT_PATH}. Using default PIL font.")
        except Exception as e:
            print(f"[WARNING] Error loading custom font: {e}. Using default PIL font.")
            self.font_main = ImageFont.load_default()
            self.font_sub = ImageFont.load_default()
            self.font_small = ImageFont.load_default()

        self.font_main_height = self.font_main.getbbox("Tg")[3] - self.font_main.getbbox("Tg")[1]
        self.font_sub_height = self.font_sub.getbbox("Tg")[3] - self.font_sub.getbbox("Tg")[1]
        self.font_small_height = self.font_small.getbbox("Tg")[3] - self.font_small.getbbox("Tg")[1]

    def _draw_rounded_rectangle(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], radius: int, fill=None, outline=None, width=1):
        """Draw a rectangle with rounded corners."""
        draw.rounded_rectangle([xy[0], xy[1], xy[2], xy[3]], radius=radius, fill=fill, outline=outline, width=width)

    def _draw_hud_box(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], fill: Tuple[int, int, int, int], outline: Tuple[int, int, int, int], thickness: int, corner_radius: int):
        """Draws a solid filled box with rounded corners and an outline."""
        self._draw_rounded_rectangle(draw, xy, corner_radius, fill=fill, outline=outline, width=thickness)

    def _draw_hud_text(self, draw: ImageDraw.ImageDraw, text: str, position: Tuple[int, int], font: ImageFont.FreeTypeFont, text_color: Tuple[int, int, int, int], outline_color: Tuple[int, int, int, int] = TEXT_OUTLINE_COLOR, outline_width: int = TEXT_OUTLINE_WIDTH):
        """
        Draws text on the HUD with optional outline for better visibility.
        Uses global TEXT_OUTLINE_COLOR and TEXT_OUTLINE_WIDTH by default.
        """
        if outline_color and outline_width > 0:
            # Draw outline by drawing text multiple times with offset
            for x_offset in range(-outline_width, outline_width + 1):
                for y_offset in range(-outline_width, outline_width + 1):
                    if x_offset != 0 or y_offset != 0:
                        draw.text((position[0] + x_offset, position[1] + y_offset), text, fill=outline_color, font=font)
        draw.text(position, text, fill=text_color, font=font)

    def _draw_glowing_line(self, draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int, int], base_width: int, glow_strength: int = 3):
        """Draws a line with a simulated glow effect."""
        for i in range(glow_strength, 0, -1):
            alpha = int(color[3] * (i / glow_strength) * 0.5) 
            draw.line((x1, y1, x2, y2), fill=color[:3] + (alpha,), width=base_width + i * 2)
        draw.line((x1, y1, x2, y2), fill=color, width=base_width)

    def _draw_wireframe_element(self, draw: ImageDraw.ImageDraw, animation_frame: int):
        """Draws abstract wireframe elements with subtle animation, adapted for UI scale."""
        scaled_thickness = max(1, int(HUD_OUTLINE_WIDTH_BASE * self.global_ui_scale))

        corner_line_length_h = int(self.frame_width * 0.04) 
        corner_line_length_v = int(self.frame_height * 0.06) 

        coords = [
            (0, corner_line_length_v, corner_line_length_h, 0), 
            (0, corner_line_length_v * 2, corner_line_length_h * 2, 0),
            (self.frame_width, corner_line_length_v, self.frame_width - corner_line_length_h, 0), 
            (self.frame_width, corner_line_length_v * 2, self.frame_width - corner_line_length_h * 2, 0),
            (0, self.frame_height - corner_line_length_v, corner_line_length_h, self.frame_height), 
            (0, self.frame_height - corner_line_length_v * 2, corner_line_length_h * 2, self.frame_height),
            (self.frame_width, self.frame_height - corner_line_length_v, self.frame_width - corner_line_length_h, self.frame_height), 
            (self.frame_width, self.frame_height - corner_line_length_v * 2, self.frame_width - corner_line_length_h * 2, self.frame_height)
        ]
        for x1, y1, x2, y2 in coords:
            self._draw_glowing_line(draw, x1, y1, x2, y2, HUD_BLUE_LIGHT, scaled_thickness)

        grid_alpha = int(HUD_BLUE_LIGHT[3] * 0.2) 
        grid_color = HUD_BLUE_LIGHT[:3] + (grid_alpha,)
        
        num_h_lines = max(3, int(self.frame_height / (150 * self.global_ui_scale))) 
        for i in range(num_h_lines):
            y_offset_base = (animation_frame % 200) * (50.0 / 200.0) 
            y_pos = int((i * (self.frame_height / num_h_lines)) + y_offset_base) % self.frame_height
            self._draw_glowing_line(draw, 0, y_pos, self.frame_width, y_pos, grid_color, max(1, scaled_thickness // 2))

        num_v_lines = max(3, int(self.frame_width / (150 * self.global_ui_scale))) 
        for i in range(num_v_lines):
            x_offset_base = (animation_frame % 200) * (50.0 / 200.0)
            x_pos = int((i * (self.frame_width / num_v_lines)) + x_offset_base) % self.frame_width
            self._draw_glowing_line(draw, x_pos, 0, x_pos, self.frame_height, grid_color, max(1, scaled_thickness // 2))

        pulse_radius_max = min(self.frame_width, self.frame_height) // 4
        pulse_radius = int(pulse_radius_max * (math.sin(animation_frame * 0.05) * 0.5 + 0.5)) 
        pulse_alpha = int(HUD_BLUE_LIGHT[3] * (1 - (pulse_radius / pulse_radius_max if pulse_radius_max > 0 else 0)) * 0.7) 
        pulse_color = HUD_BLUE_LIGHT[:3] + (pulse_alpha,)
        self._draw_rounded_rectangle(draw, [self.frame_width//2 - pulse_radius, self.frame_height//2 - pulse_radius, 
                      self.frame_width//2 + pulse_radius, self.frame_height//2 + pulse_radius], 
                     radius=pulse_radius, outline=pulse_color, width=max(1, scaled_thickness))
        
    def _draw_bar_graph(self, draw: ImageDraw.ImageDraw, x: int, y: int, width: int, height: int, values: List[Dict[str, Any]]):
        """Draws a simple bar graph for confidences, adapted for UI scale."""
        bar_spacing = max(1, int(width * 0.01)) 
        
        font_bbox = self.font_small.getbbox("Tg")
        font_height = font_bbox[3] - font_bbox[1]

        max_bar_height = height - (font_height * 2) - (bar_spacing * 2)
        max_bar_height = max(1, max_bar_height) 

        num_bars = len(values)
        if num_bars == 0: return

        individual_bar_width = (width - (num_bars + 1) * bar_spacing) // num_bars
        individual_bar_width = max(1, individual_bar_width) 

        for i, pred_dict in enumerate(values):
            label = pred_dict["label"]
            value = pred_dict["confidence"]

            bar_height_actual = int(max_bar_height * value)
            bar_x1 = x + bar_spacing + i * (individual_bar_width + bar_spacing)
            bar_y1 = y + height - bar_height_actual - font_height - bar_spacing
            bar_x2 = bar_x1 + individual_bar_width
            bar_y2 = y + height - font_height - bar_spacing

            bar_color = SCENE_LABEL_COLORS.get(label, HUD_CYAN_LIGHT)
            self._draw_rounded_rectangle(draw, (bar_x1, bar_y1, bar_x2, bar_y2), max(1, int(3 * self.global_ui_scale)), fill=bar_color+(150,), outline=bar_color, width=1)
            
            self._draw_hud_text(draw, f"{value:.2f}", (bar_x1, bar_y1 - font_height - max(1, int(2 * self.global_ui_scale))), self.font_small, HUD_TEXT_COLOR_PRIMARY)
            
            display_label = label.replace('_', ' ')
            label_bbox = self.font_small.getbbox(display_label)
            text_w = label_bbox[2] - label_bbox[0]
            
            if text_w > individual_bar_width:
                avg_char_width = text_w / len(display_label) if len(display_label) > 0 else 1
                chars_to_fit = int(individual_bar_width / avg_char_width) - 1 
                if chars_to_fit > 0:
                    display_label = display_label[:max(chars_to_fit, 1)].strip() + "." 
                else:
                    display_label = "" 
            
            self._draw_hud_text(draw, display_label, (bar_x1, bar_y2 + max(1, int(2 * self.global_ui_scale))), self.font_small, HUD_TEXT_COLOR_SECONDARY)

    def render_hud(self, frame: np.ndarray, display_objects: List[Dict[str, Any]], 
                    scene_analysis_data: Dict[str, Any], system_status: Dict[str, Any],
                    event_log_history: deque, all_detected_plates: deque,
                    animation_frame: int, plate_lookup_details: Dict[str, str] = None):
        """
        Renders the complete Iron Man HUD overlay onto the given frame.
        """
        hud_layer = Image.new('RGBA', (self.frame_width, self.frame_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(hud_layer)

        # --- Draw Wireframe Overlay (Behind other elements, with animation) ---
        self._draw_wireframe_element(draw, animation_frame)

        # --- Advanced Lane & Road Boundary Detection (Feature 8) ---
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv2.Canny(blurred_frame, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
                if 10 < abs(angle) < 170: 
                    self._draw_glowing_line(draw, x1, y1, x2, y2, LANE_LINE_COLOR, base_width=max(1, int(LANE_LINE_THICKNESS * self.global_ui_scale)))


        # --- Determine dynamic UI properties ---
        dynamic_padding = max(5, int(HUD_PADDING_BASE * self.global_ui_scale))
        hud_outline_width = max(1, int(HUD_OUTLINE_WIDTH_BASE * self.global_ui_scale))
        hud_corner_radius = max(5, int(HUD_CORNER_RADIUS_BASE * self.global_ui_scale))

        panel_width = max(int(self.frame_width * PANEL_BASE_WIDTH_RATIO), int(150 * self.global_ui_scale)) 
        panel_height = max(int(self.frame_height * PANEL_BASE_HEIGHT_RATIO), int(100 * self.global_ui_scale)) 
        
        # --- Top-left HUD block: Scene, Status, and Action ---
        panel1_x = dynamic_padding
        panel1_y = dynamic_padding
        panel1_actual_height = panel_height * 0.9 # Slightly shorter to fit more

        self._draw_hud_box(draw, (panel1_x, panel1_y, panel1_x + panel_width, panel1_y + panel1_actual_height), HUD_BLUE_DARK_TRANSPARENT, HUD_BLUE_LIGHT, hud_outline_width, hud_corner_radius)
        
        current_y_in_panel1 = panel1_y + int(10 * self.global_ui_scale) 

        scene_color_for_display = SCENE_LABEL_COLORS.get(scene_analysis_data['most_common_scene'], HUD_TEXT_COLOR_PRIMARY)
        scene_display_text = f"SCENE: {scene_analysis_data['most_common_scene'].replace('_', ' ').upper()}"
        
        pulse_alpha = int(255 * (math.sin(animation_frame * 0.1) * 0.2 + 0.8)) 
        text_color_pulsating = scene_color_for_display[:3] + (pulse_alpha,)

        self._draw_hud_text(draw, scene_display_text, (panel1_x + int(20 * self.global_ui_scale), current_y_in_panel1), self.font_main, text_color_pulsating)
        current_y_in_panel1 += self.font_main_height + int(5 * self.global_ui_scale)
        self._draw_hud_text(draw, f"CONF: {scene_analysis_data['smoothed_scene_confidence']:.2f}", (panel1_x + int(20 * self.global_ui_scale), current_y_in_panel1), self.font_sub, HUD_TEXT_COLOR_HIGHLIGHT)
        
        line_start_x = panel1_x + int(15 * self.global_ui_scale)
        line_end_x = panel1_x + panel_width - int(15 * self.global_ui_scale)
        line_y = current_y_in_panel1 + self.font_sub_height + int(15 * self.global_ui_scale)
        self._draw_glowing_line(draw, line_start_x, line_y, line_end_x, line_y, HUD_CYAN_LIGHT, base_width=max(1, int(2 * self.global_ui_scale)))
        
        scan_x = line_start_x + int((line_end_x - line_start_x) * (animation_frame % 60 / 60.0))
        self._draw_glowing_line(draw, scan_x, line_y - int(5 * self.global_ui_scale), scan_x, line_y + int(5 * self.global_ui_scale), HUD_CYAN_LIGHT, base_width=max(1, int(1 * self.global_ui_scale)))

        current_y_in_panel1 = line_y + int(10 * self.global_ui_scale)

        alert_text_color = HUD_TEXT_COLOR_PRIMARY
        
        if scene_analysis_data['current_alert_level'] == "WARNING":
            alert_text_color = SCENE_LABEL_COLORS["dense_traffic"] 
        elif scene_analysis_data['current_alert_level'] in ["CRITICAL_ALERT", "ALERT_SENT"]:
            alert_text_color = SCENE_LABEL_COLORS["accident"] 
            pulsating_fill_alpha = int(100 * (math.sin(animation_frame * 0.3) * 0.5 + 0.5))
            pulsating_fill_color = (HUD_RED_CRITICAL[0], HUD_RED_CRITICAL[1], HUD_RED_CRITICAL[2], pulsating_fill_alpha)
            
            draw.rectangle((0, 0, self.frame_width-1, self.frame_height-1), outline=HUD_RED_CRITICAL, width=max(2, int(self.frame_width * 0.005)), 
                           fill=pulsating_fill_color)
                           

        self._draw_hud_text(draw, f"STATUS: {scene_analysis_data['current_alert_level'].replace('_', ' ').upper()}", (panel1_x + int(20 * self.global_ui_scale), current_y_in_panel1), self.font_sub, alert_text_color)
        current_y_in_panel1 += self.font_sub_height + int(5 * self.global_ui_scale)
        self._draw_hud_text(draw, scene_analysis_data['display_action_message'], (panel1_x + int(20 * self.global_ui_scale), current_y_in_panel1), self.font_small, HUD_TEXT_COLOR_SECONDARY)


        # --- Top-Right Panel: Object Classification ---
        panel2_x = self.frame_width - panel_width - dynamic_padding
        panel2_y = dynamic_padding

        self._draw_hud_box(draw, (panel2_x, panel2_y, panel2_x + panel_width, panel2_y + panel_height), HUD_BLUE_DARK_TRANSPARENT, HUD_CYAN_LIGHT, hud_outline_width, hud_corner_radius)
        
        title_text_obj = "OBJECT CLASSIFICATION"
        self._draw_hud_text(draw, title_text_obj, (panel2_x + int(20 * self.global_ui_scale), panel2_y + int(15 * self.global_ui_scale)), self.font_sub, HUD_TEXT_COLOR_PRIMARY)
        self._draw_glowing_line(draw, panel2_x + int(20 * self.global_ui_scale), panel2_y + int(50 * self.global_ui_scale), panel2_x + panel_width - int(20 * self.global_ui_scale), panel2_y + int(50 * self.global_ui_scale), HUD_CYAN_LIGHT, base_width=max(1, int(1 * self.global_ui_scale)))

        object_counts_display: Dict[str, int] = {}
        for obj in display_objects: 
            object_counts_display[obj['label']] = object_counts_display.get(obj['label'], 0) + 1

        obj_content_y_start = panel2_y + int(60 * self.global_ui_scale)
        obj_line_height = self.font_small_height + int(5 * self.global_ui_scale)
        
        available_height_for_obj_list = panel_height - (obj_content_y_start - panel2_y) - int(10 * self.global_ui_scale)
        max_lines_obj = max(1, available_height_for_obj_list // obj_line_height)

        current_obj_lines_count = 0
        sorted_objects_display = sorted(object_counts_display.items(), key=lambda item: item[1], reverse=True) 
        
        self._draw_hud_text(draw, "Overall:", (panel2_x + int(20 * self.global_ui_scale), obj_content_y_start), self.font_small, HUD_TEXT_COLOR_HIGHLIGHT)
        current_obj_lines_count += 1
        
        for obj_label, count in sorted_objects_display:
            if current_obj_lines_count < max_lines_obj - 1: 
                display_obj_text = f"- {obj_label.capitalize()}: {count}"
                self._draw_hud_text(draw, display_obj_text, (panel2_x + int(20 * self.global_ui_scale), obj_content_y_start + current_obj_lines_count * obj_line_height), self.font_small, HUD_TEXT_COLOR_SECONDARY)
                current_obj_lines_count += 1
        
        total_detected_objects = sum(object_counts_display.values()) 
        if current_obj_lines_count < max_lines_obj: 
             self._draw_hud_text(draw, f"TOTAL: {total_detected_objects}", (panel2_x + int(20 * self.global_ui_scale), obj_content_y_start + current_obj_lines_count * obj_line_height), self.font_small, HUD_TEXT_COLOR_HIGHLIGHT)


        # --- Middle-Right Panel: Vehicle Info (SIMULATED) ---
        panel3_x = panel2_x
        panel3_y = panel2_y + panel_height + dynamic_padding 

        self._draw_hud_box(draw, (panel3_x, panel3_y, panel3_x + panel_width, panel3_y + panel_height), HUD_BLUE_DARK_TRANSPARENT, HUD_YELLOW_ACCENT, hud_outline_width, hud_corner_radius)
        
        title_text_npr = "VEHICLE INFO (SIMULATED)" 
        self._draw_hud_text(draw, title_text_npr, (panel3_x + int(20 * self.global_ui_scale), panel3_y + int(15 * self.global_ui_scale)), self.font_sub, HUD_TEXT_COLOR_PRIMARY)
        self._draw_glowing_line(draw, panel3_x + int(20 * self.global_ui_scale), panel3_y + int(50 * self.global_ui_scale), panel3_x + panel_width - int(20 * self.global_ui_scale), panel3_y + int(50 * self.global_ui_scale), HUD_YELLOW_ACCENT, base_width=max(1, int(1 * self.global_ui_scale)))

        npr_content_y_start = panel3_y + int(60 * self.global_ui_scale)
        npr_line_height = self.font_small_height + int(5 * self.global_ui_scale)
        available_height_for_npr_list = panel_height - (npr_content_y_start - panel3_y) - int(10 * self.global_ui_scale)
        max_lines_npr = max(1, available_height_for_npr_list // npr_line_height)

        current_npr_lines_count = 0
        found_plate_data = False
        for obj in display_objects:
            if obj['label'] in ["car", "truck"] and obj['plate_number']:
                self._draw_hud_text(draw, f"Plate: {obj['plate_number']}", (panel3_x + int(20 * self.global_ui_scale), npr_content_y_start), self.font_small, HUD_TEXT_COLOR_HIGHLIGHT)
                current_npr_lines_count += 1
                
                for key, value in obj['plate_data'].items():
                    if current_npr_lines_count < max_lines_npr:
                        self._draw_hud_text(draw, f"- {key}: {value}", (panel3_x + int(20 * self.global_ui_scale), npr_content_y_start + current_npr_lines_count * npr_line_height), self.font_small, HUD_TEXT_COLOR_SECONDARY)
                        current_npr_lines_count += 1
                found_plate_data = True
                break 
        
        if not found_plate_data:
            self._draw_hud_text(draw, "Scanning for vehicles...", (panel3_x + int(20 * self.global_ui_scale), npr_content_y_start), self.font_small, HUD_TEXT_COLOR_SECONDARY)


        # --- Bottom-Right Panel: Object Details, System Health, Scene Confidence ---
        panel4_x = panel2_x
        panel4_y = panel3_y + panel_height + dynamic_padding 
        panel4_actual_height = self.frame_height - panel4_y - dynamic_padding - (max(int(self.frame_height * 0.04), int(30 * self.global_ui_scale))) # Adjust for bottom bar

        self._draw_hud_box(draw, (panel4_x, panel4_y, panel4_x + panel_width, panel4_y + panel4_actual_height), HUD_BLUE_DARK_TRANSPARENT, HUD_GREEN_LIGHT, hud_outline_width, hud_corner_radius)
        
        title_text_obj_det = "OBJECT DETAILS"
        self._draw_hud_text(draw, title_text_obj_det, (panel4_x + int(20 * self.global_ui_scale), panel4_y + int(15 * self.global_ui_scale)), self.font_sub, HUD_TEXT_COLOR_PRIMARY)
        self._draw_glowing_line(draw, panel4_x + int(20 * self.global_ui_scale), panel4_y + int(50 * self.global_ui_scale), panel4_x + panel_width - int(20 * self.global_ui_scale), panel4_y + int(50 * self.global_ui_scale), HUD_CYAN_LIGHT, base_width=max(1, int(1 * self.global_ui_scale)))

        obj_det_content_y_start = panel4_y + int(60 * self.global_ui_scale)
        obj_det_line_height = self.font_small_height + int(5 * self.global_ui_scale)
        
        max_lines_obj_det = 3 
        
        current_obj_det_lines_count = 0
        sorted_display_objects = sorted(display_objects, key=lambda x: x['threat_score'], reverse=True)

        for i, obj in enumerate(sorted_display_objects):
            if current_obj_det_lines_count < max_lines_obj_det * 3: # Each object takes 3 lines
                speed_px_per_frame = math.sqrt(obj['velocity_x']**2 + obj['velocity_y']**2)
                bbox_area = (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3] - obj['bbox'][1])
                relative_distance_val = 0
                if bbox_area > 0:
                    relative_distance_val = 1.0 / (math.sqrt(bbox_area) / (self.frame_width * self.global_ui_scale) + 0.0001)
                
                relative_distance_display = f"{min(relative_distance_val, 999.9):.1f}u" 
                speed_display = f"{speed_px_per_frame:.1f}p/f" 

                detail_text = f"ID:{obj['id']} {obj['label']} | T:{obj['threat_score']:.0f}"
                self._draw_hud_text(draw, detail_text, (panel4_x + int(20 * self.global_ui_scale), obj_det_content_y_start + current_obj_det_lines_count * obj_det_line_height), self.font_small, HUD_TEXT_COLOR_SECONDARY)
                current_obj_det_lines_count += 1
                
                speed_dist_text = f"  Spd:{speed_display} Dist:{relative_distance_display}"
                self._draw_hud_text(draw, speed_dist_text, (panel4_x + int(20 * self.global_ui_scale), obj_det_content_y_start + current_obj_det_lines_count * obj_det_line_height), self.font_small, HUD_TEXT_COLOR_SECONDARY)
                current_obj_det_lines_count += 1
                
                if obj['label'] in ["car", "truck", "bus"]:
                    sig_text = f"  Sig: {obj['vehicle_signature'].get('Make', 'N/A')}, {obj['vehicle_signature'].get('Color', 'N/A')}"
                    self._draw_hud_text(draw, sig_text, (panel4_x + int(20 * self.global_ui_scale), obj_det_content_y_start + current_obj_det_lines_count * obj_det_line_height), self.font_small, HUD_TEXT_COLOR_SECONDARY)
                    current_obj_det_lines_count += 1


        if not display_objects:
            self._draw_hud_text(draw, "No objects for detailed view.", (panel4_x + int(20 * self.global_ui_scale), obj_det_content_y_start), self.font_small, HUD_TEXT_COLOR_SECONDARY)
            current_obj_det_lines_count += 3 

        line_y_sys_sep = obj_det_content_y_start + current_obj_det_lines_count * obj_det_line_height + int(10 * self.global_ui_scale)
        self._draw_glowing_line(draw, panel4_x + int(20 * self.global_ui_scale), line_y_sys_sep, panel4_x + panel_width - int(20 * self.global_ui_scale), line_y_sys_sep, HUD_GREEN_LIGHT, base_width=max(1, int(1 * self.global_ui_scale)))

        sys_content_y_start = line_y_sys_sep + int(10 * self.global_ui_scale)
        self._draw_hud_text(draw, "SYSTEM HEALTH", (panel4_x + int(20 * self.global_ui_scale), sys_content_y_start), self.font_sub, HUD_TEXT_COLOR_PRIMARY)
        sys_content_y_start += self.font_sub_height + int(5 * self.global_ui_scale)

        sys_lines = [
            f"Frames: {system_status['frame_count']}",
            f"FPS: {system_status['fps']:.1f}",
            f"CPU Load: {system_status['cpu_load']:.1f}%",
            f"GPU Load: {system_status['gpu_load']:.1f}%",
            f"Data Rate: {system_status['data_rate']:.1f} MB/s",
            f"Device: {system_status['device']}"
        ]
        
        sys_line_height = self.font_small_height + int(5 * self.global_ui_scale)
        current_sys_lines_count = 0
        available_height_for_sys_list = panel4_actual_height - (sys_content_y_start - panel4_y) - int(10 * self.global_ui_scale)
        max_sys_lines = max(1, available_height_for_sys_list // sys_line_height)

        for i, line in enumerate(sys_lines):
            if i < max_sys_lines: 
                self._draw_hud_text(draw, line, (panel4_x + int(20 * self.global_ui_scale), sys_content_y_start + i * sys_line_height), self.font_small, HUD_TEXT_COLOR_SECONDARY)
                current_sys_lines_count += 1

        graph_title_bbox = self.font_small.getbbox("SCENE CONFIDENCE:") 
        graph_title_height_actual = graph_title_bbox[3] - graph_title_bbox[1] 
        
        remaining_height_for_graph = panel4_actual_height - (sys_content_y_start + current_sys_lines_count * sys_line_height - panel4_y) - int(10 * self.global_ui_scale) - graph_title_height_actual
        graph_height_actual = max(int(self.frame_height * 0.08), remaining_height_for_graph) 

        graph_y_start = panel4_y + panel4_actual_height - graph_height_actual - int(10 * self.global_ui_scale)
        
        self._draw_hud_text(draw, "SCENE CONFIDENCE:", (panel4_x + int(20 * self.global_ui_scale), graph_y_start - graph_title_height_actual), self.font_small, HUD_TEXT_COLOR_PRIMARY)
        self._draw_bar_graph(draw, panel4_x + int(10 * self.global_ui_scale), graph_y_start, panel_width - int(20 * self.global_ui_scale), graph_height_actual, scene_analysis_data['all_predictions'])


        # --- Bottom-Left Panel: Event Log ---
        panel5_x = dynamic_padding
        panel5_y = self.frame_height - panel_height - dynamic_padding - (max(int(self.frame_height * 0.04), int(30 * self.global_ui_scale))) 
        
        self._draw_hud_box(draw, (panel5_x, panel5_y, panel5_x + panel_width, panel5_y + panel_height), HUD_BLUE_DARK_TRANSPARENT, HUD_BLUE_LIGHT, hud_outline_width, hud_corner_radius)
        self._draw_hud_text(draw, "EVENT LOG", (panel5_x + int(20 * self.global_ui_scale), panel5_y + int(15 * self.global_ui_scale)), self.font_sub, HUD_TEXT_COLOR_PRIMARY)
        self._draw_glowing_line(draw, panel5_x + int(20 * self.global_ui_scale), panel5_y + int(50 * self.global_ui_scale), panel5_x + panel_width - int(20 * self.global_ui_scale), panel5_y + int(50 * self.global_ui_scale), HUD_BLUE_LIGHT, base_width=max(1, int(1 * self.global_ui_scale)))

        log_content_y_start = panel5_y + int(60 * self.global_ui_scale)
        log_line_height = self.font_small_height + int(5 * self.global_ui_scale) 
        available_height_for_log = panel_height - (log_content_y_start - panel5_y) - int(10 * self.global_ui_scale)
        max_log_lines = max(1, available_height_for_log // log_line_height)

        if len(event_log_history) > 0:
            effective_log_length = len(event_log_history)
            scroll_speed_factor = max(1.0, effective_log_length / max_log_lines) if max_log_lines > 0 else 1.0
            scroll_duration_frames = int(effective_log_length * 30 / scroll_speed_factor) 
            
            if scroll_duration_frames == 0: scroll_denominator = 1 
            else: scroll_denominator = scroll_duration_frames

            log_scroll_pos_norm = (animation_frame % scroll_denominator) / scroll_denominator
            
            total_scroll_lines = (effective_log_length - max_log_lines) if effective_log_length > max_log_lines else 0
            current_scroll_offset_lines = total_scroll_lines * log_scroll_pos_norm
            
            for i in range(max_log_lines):
                log_entry_target_index = i + current_scroll_offset_lines
                actual_log_index = int(log_entry_target_index)
                
                if actual_log_index < effective_log_length and actual_log_index >= 0:
                    log_entry = event_log_history[actual_log_index]
                    
                    y_pos_adjustment = (log_entry_target_index - actual_log_index) * log_line_height
                    y_pos = log_content_y_start + i * log_line_height - y_pos_adjustment
                    
                    display_log_text = log_entry
                    self._draw_hud_text(draw, display_log_text, (panel5_x + int(20 * self.global_ui_scale), y_pos), self.font_small, HUD_TEXT_COLOR_SECONDARY)
        else: 
            self._draw_hud_text(draw, "No events to display.", (panel5_x + int(20 * self.global_ui_scale), log_content_y_start), self.font_small, HUD_TEXT_COLOR_SECONDARY)


        # --- Bottom Status Bar (Temperature, Time) ---
        status_bar_height = max(int(self.frame_height * 0.04), int(30 * self.global_ui_scale))
        status_bar_y = self.frame_height - status_bar_height
        draw.rectangle((0, status_bar_y, self.frame_width, self.frame_height), fill=HUD_BLUE_DARK_TRANSPARENT, outline=HUD_BLUE_LIGHT, width=hud_outline_width)
        
        temp_text = "28°C Partly Cloudy" 
        time_text = datetime.now().strftime('%H:%M:%S')
        date_text = datetime.now().strftime('%d-%m-%Y')

        self._draw_hud_text(draw, temp_text, (dynamic_padding, status_bar_y + int(5 * self.global_ui_scale)), self.font_small, HUD_TEXT_COLOR_SECONDARY)
        self._draw_hud_text(draw, time_text, (self.frame_width - dynamic_padding - (self.font_small.getbbox(time_text)[2] - self.font_small.getbbox(time_text)[0]), status_bar_y + int(5 * self.global_ui_scale)), self.font_small, HUD_TEXT_COLOR_SECONDARY)
        self._draw_hud_text(draw, date_text, (self.frame_width - dynamic_padding - (self.font_small.getbbox(date_text)[2] - self.font_small.getbbox(date_text)[0]), status_bar_y + int(5 * self.global_ui_scale) + self.font_small_height + int(2 * self.global_ui_scale)), self.font_small, HUD_TEXT_COLOR_SECONDARY)

        # --- Number Plate Log Panel (Bottom Right) ---
        panel6_x = self.frame_width - panel_width - dynamic_padding
        panel6_y = self.frame_height - panel_height - dynamic_padding - (max(int(self.frame_height * 0.04), int(30 * self.global_ui_scale))) 

        self._draw_hud_box(draw, (panel6_x, panel6_y, panel6_x + panel_width, panel6_y + panel_height), HUD_BLUE_DARK_TRANSPARENT, HUD_YELLOW_ACCENT, hud_outline_width, hud_corner_radius)
        self._draw_hud_text(draw, "PLATE LOG (SIMULATED)", (panel6_x + int(20 * self.global_ui_scale), panel6_y + int(15 * self.global_ui_scale)), self.font_sub, HUD_TEXT_COLOR_PRIMARY)
        self._draw_glowing_line(draw, panel6_x + int(20 * self.global_ui_scale), panel6_y + int(50 * self.global_ui_scale), panel6_x + panel_width - int(20 * self.global_ui_scale), panel6_y + int(50 * self.global_ui_scale), HUD_YELLOW_ACCENT, base_width=max(1, int(1 * self.global_ui_scale)))

        plate_log_content_y_start = panel6_y + int(60 * self.global_ui_scale)
        plate_log_line_height = self.font_small_height + int(5 * self.global_ui_scale)
        available_height_for_plate_log = panel_height - (plate_log_content_y_start - panel6_y) - int(10 * self.global_ui_scale)
        max_plate_log_lines = max(1, available_height_for_plate_log // plate_log_line_height)

        if len(all_detected_plates) > 0:
            effective_plate_log_length = len(all_detected_plates)
            scroll_speed_factor_plates = max(1.0, effective_plate_log_length / max_plate_log_lines) if max_plate_log_lines > 0 else 1.0
            scroll_duration_frames_plates = int(effective_plate_log_length * 30 / scroll_speed_factor_plates) 
            
            if scroll_duration_frames_plates == 0: scroll_denominator_plates = 1 
            else: scroll_denominator_plates = scroll_duration_frames_plates

            plate_log_scroll_pos_norm = (animation_frame % scroll_denominator_plates) / scroll_denominator_plates
            
            total_scroll_lines_plates = (effective_plate_log_length - max_plate_log_lines) if effective_plate_log_length > max_plate_log_lines else 0
            current_scroll_offset_lines_plates = total_scroll_lines_plates * plate_log_scroll_pos_norm
            
            for i in range(max_plate_log_lines):
                plate_log_entry_target_index = i + current_scroll_offset_lines_plates
                actual_plate_log_index = int(plate_log_entry_target_index)
                
                if actual_plate_log_index < effective_plate_log_length and actual_plate_log_index >= 0:
                    plate_entry = all_detected_plates[actual_plate_log_index]
                    
                    y_pos_adjustment = (plate_log_entry_target_index - actual_plate_log_index) * plate_log_line_height
                    y_pos = plate_log_content_y_start + i * plate_log_line_height - y_pos_adjustment
                    
                    display_plate_text = plate_entry
                    self._draw_hud_text(draw, display_plate_text, (panel6_x + int(20 * self.global_ui_scale), y_pos), self.font_small, HUD_TEXT_COLOR_SECONDARY)
        else: 
            self._draw_hud_text(draw, "No plates detected yet.", (panel6_x + int(20 * self.global_ui_scale), plate_log_content_y_start), self.font_small, HUD_TEXT_COLOR_SECONDARY)

        # --- Draw Plate Lookup Details Pop-up (if active) ---
        if plate_lookup_details:
            popup_width = int(self.frame_width * 0.4)
            popup_height = int(self.frame_height * 0.6)
            popup_x = (self.frame_width - popup_width) // 2
            popup_y = (self.frame_height - popup_height) // 2

            self._draw_hud_box(draw, (popup_x, popup_y, popup_x + popup_width, popup_y + popup_height), 
                               HUD_BLUE_DARK_TRANSPARENT, HUD_CYAN_LIGHT, hud_outline_width * 2, hud_corner_radius * 2)
            
            self._draw_hud_text(draw, "PLATE DETAILS (SIMULATED)", (popup_x + int(20 * self.global_ui_scale), popup_y + int(15 * self.global_ui_scale)), self.font_sub, HUD_TEXT_COLOR_HIGHLIGHT)
            self._draw_glowing_line(draw, popup_x + int(20 * self.global_ui_scale), popup_y + int(50 * self.global_ui_scale), popup_x + popup_width - int(20 * self.global_ui_scale), popup_y + int(50 * self.global_ui_scale), HUD_CYAN_LIGHT, base_width=max(1, int(1 * self.global_ui_scale)))

            detail_y_start = popup_y + int(60 * self.global_ui_scale)
            detail_line_height = self.font_small_height + int(5 * self.global_ui_scale)
            
            for i, (key, value) in enumerate(plate_lookup_details.items()):
                self._draw_hud_text(draw, f"{key}: {value}", (popup_x + int(20 * self.global_ui_scale), detail_y_start + i * detail_line_height), self.font_small, HUD_TEXT_COLOR_PRIMARY)
            
            self._draw_hud_text(draw, "Press 'Esc' to close", (popup_x + int(20 * self.global_ui_scale), popup_y + popup_height - int(30 * self.global_ui_scale)), self.font_small, HUD_TEXT_COLOR_HIGHLIGHT)


        # --- Draw tracked objects (bounding boxes, labels, metadata, trajectory) ---
        for obj in display_objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            label = obj['label']
            conf = obj['confidence']
            obj_id = obj['id'] 
            vx, vy = obj['velocity_x'], obj['velocity_y']
            threat_score = obj['threat_score']
            is_collision_alert = obj.get('is_collision_alert', False) 

            if is_collision_alert:
                flash_alpha = int(255 * (math.sin(animation_frame * 0.5) * 0.5 + 0.5))
                color = HUD_RED_CRITICAL[:3]
                color_with_alpha = color + (flash_alpha,)
            elif threat_score >= 80:
                color = HUD_RED_CRITICAL[:3] 
                color_with_alpha = color + (180,)
            elif threat_score >= 50:
                color = HUD_YELLOW_ACCENT[:3] 
                color_with_alpha = color + (180,)
            else:
                color = (BBOX_COLORS.get(label, (200, 200, 200))) 
                color_with_alpha = color + (180,)
            
            draw.rectangle([x1, y1, x2, y2], outline=color_with_alpha, width=max(1, int(self.frame_width * 0.002 * self.global_ui_scale))) 
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            velocity_magnitude = math.sqrt(vx**2 + vy**2)
            if velocity_magnitude > 0:
                norm_vx = (vx / velocity_magnitude) * TRAJECTORY_PREDICTION_LENGTH * self.global_ui_scale
                norm_vy = (vy / velocity_magnitude) * TRAJECTORY_PREDICTION_LENGTH * self.global_ui_scale
            else:
                norm_vx, norm_vy = 0, 0 

            traj_end_x = int(center_x + norm_vx)
            traj_end_y = int(center_y + norm_vy)
            
            self._draw_glowing_line(draw, int(center_x), int(center_y), traj_end_x, traj_end_y, color + (100,), base_width=max(1, int(2 * self.global_ui_scale)))


            speed_px_per_frame = math.sqrt(vx**2 + vy**2)
            bbox_area = (x2 - x1) * (y2 - y1)
            relative_distance_val = 0
            if bbox_area > 0:
                relative_distance_val = 1.0 / (math.sqrt(bbox_area) / (self.frame_width * self.global_ui_scale) + 0.0001)
            
            relative_distance_display = f"{min(relative_distance_val, 999.9):.1f}u" 
            speed_display = f"{speed_px_per_frame:.1f}p/f" 

            text_label = f"{label.upper()} ({conf:.2f}) ID:{obj_id} T:{threat_score:.0f}" 
            bbox_font_small = self.font_small
            
            bbox_label_metrics = bbox_font_small.getbbox(text_label)
            text_w = bbox_label_metrics[2] - bbox_label_metrics[0]
            text_h = bbox_label_metrics[3] - bbox_label_metrics[1]
            
            max_bbox_label_width = x2 - x1 
            if max_bbox_label_width < int(50 * self.global_ui_scale): 
                max_bbox_label_width = int(50 * self.global_ui_scale) 
            
            if text_w > max_bbox_label_width:
                chars_to_fit = int(len(text_label) * (max_bbox_label_width / (text_w if text_w > 0 else 1.0))) - 2 
                if chars_to_fit > 0:
                    text_label = text_label[:max(chars_to_fit, 1)].strip()
                    if len(text_label) > 1:
                        text_label += "." 
                else:
                    text_label = "" 


            text_x = x1 + max(1, int(4 * self.global_ui_scale))
            text_y = y1 - text_h - max(1, int(6 * self.global_ui_scale))
            if text_y < 0: text_y = y1 + max(1, int(2 * self.global_ui_scale)) 
                
            self._draw_rounded_rectangle(draw, [text_x - max(1, int(2 * self.global_ui_scale)), text_y - max(1, int(2 * self.global_ui_scale)), text_x + text_w + max(1, int(4 * self.global_ui_scale)), text_y + text_h + max(1, int(4 * self.global_ui_scale))], radius=max(1, int(4 * self.global_ui_scale)), fill=color_with_alpha)
            self._draw_hud_text(draw, text_label, (text_x, text_y), bbox_font_small, (255,255,255,255))

        return hud_layer
