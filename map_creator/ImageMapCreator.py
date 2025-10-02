import pygame
import numpy as np
from PIL import Image
import os
import sys

# Thêm đường dẫn để import Obstacle
def get_project_root():
    return os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(get_project_root(), 'environment'))
from environment.Obstacle import Obstacle

# Cấu hình
TARGET_SIZE = 32              # Grid size sau khi scale
CELL_SIZE = 16                # Kích thước mỗi cell (pixel)
ENV_PIXEL = CELL_SIZE * TARGET_SIZE
ENV_PADDING = int(ENV_PIXEL * 0.06)
SCREEN_WIDTH = ENV_PIXEL + 2 * ENV_PADDING
SCREEN_HEIGHT = ENV_PIXEL + 3 * ENV_PADDING

def load_image_to_binary(image_path, target_size=32):
    """
    Đọc ảnh PNG và chuyển thành binary map
    Đen (obstacles) = 1, Trắng (free space) = 0
    """
    try:
        # Đọc ảnh
        img = Image.open(image_path)
        # Chuyển sang grayscale
        img = img.convert('L')
        
        # Resize về target_size x target_size
        img = img.resize((target_size, target_size))
        
        # Chuyển thành numpy array
        img_array = np.array(img)
        
        # Chuyển thành binary: < 128 = đen (obstacle), >= 128 = trắng (free)
        binary_map = (img_array < 128).astype(int)
        
        return binary_map
        
    except Exception as e:
        print(f"Lỗi khi đọc ảnh: {e}")
        return None

def binary_to_obstacles(binary_map):
    """
    Chuyển binary map thành danh sách obstacles
    """
    obstacles = []
    rows, cols = binary_map.shape
    
    for i in range(rows):
        for j in range(cols):
            if binary_map[i][j] == 1:  # Obstacle
                x = ENV_PADDING + j * CELL_SIZE + CELL_SIZE / 2
                y = ENV_PADDING + i * CELL_SIZE + CELL_SIZE / 2
                obstacles.append(Obstacle(x, y, CELL_SIZE, CELL_SIZE, True, [0, 0]))
    
    return obstacles

def select_start_goal_and_dynamic_obstacles(binary_map):
    """
    Giao diện interactive để chọn start, goal và vẽ dynamic obstacles
    """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("1. Select Start/Goal, 2. Edit Obstacles")
    font = pygame.font.SysFont("arial", max(12, ENV_PADDING // 3))
    
    # Tạo static obstacles từ binary map
    static_obstacles = binary_to_obstacles(binary_map)
    dynamic_obstacles = []
    
    start_pos = None
    goal_pos = None
    mode = "start"  # "start" -> "goal" -> "dynamic" -> "done" (thêm "static_edit")
    
    # Biến cho việc vẽ dynamic obstacles
    drawing = False
    mx = my = 0
    
    def find_obstacle_at_position(x, y, obstacle_list):
        """Tìm obstacle tại vị trí click"""
        for i, obs in enumerate(obstacle_list):
            if (obs.x - obs.width/2 <= x <= obs.x + obs.width/2 and 
                obs.y - obs.height/2 <= y <= obs.y + obs.height/2):
                return i
        return None
    
    def draw_all():
        screen.fill((255, 255, 255))
        
        # Vẽ lưới
        for k in range(1, TARGET_SIZE):
            pygame.draw.line(screen, (0, 0, 0),
                           (ENV_PADDING + k * CELL_SIZE, ENV_PADDING),
                           (ENV_PADDING + k * CELL_SIZE, ENV_PADDING + ENV_PIXEL), 1)
            pygame.draw.line(screen, (0, 0, 0),
                           (ENV_PADDING, ENV_PADDING + k * CELL_SIZE),
                           (ENV_PADDING + ENV_PIXEL, ENV_PADDING + k * CELL_SIZE), 1)
        
        # Vẽ khung ngoài
        pygame.draw.rect(screen, (0, 0, 0), (ENV_PADDING, ENV_PADDING, ENV_PIXEL, ENV_PIXEL), 3)
        
        # Vẽ static obstacles (màu đen)
        for obs in static_obstacles:
            obs.draw(screen)
        
        # Vẽ dynamic obstacles (màu xám)
        for obs in dynamic_obstacles:
            pygame.draw.rect(screen, (128, 128, 128), 
                           (int(obs.x - obs.width/2), int(obs.y - obs.height/2), 
                            int(obs.width), int(obs.height)))
        
        # Preview khi vẽ dynamic obstacle
        if drawing and mode == "dynamic":
            x2, y2 = pygame.mouse.get_pos()
            pygame.draw.rect(screen, (128, 128, 128), 
                           (min(mx, x2), min(my, y2), abs(x2 - mx), abs(y2 - my)), 1)
        
        # Vẽ start (màu xanh lá)
        if start_pos:
            pygame.draw.circle(screen, (0, 255, 0), start_pos, CELL_SIZE // 2)
        
        # Vẽ goal (màu đỏ)
        if goal_pos:
            pygame.draw.circle(screen, (255, 0, 0), goal_pos, CELL_SIZE // 2)
        
        # Hiển thị hướng dẫn
        if mode == "start":
            text = font.render("Click to place START (green)", True, (0, 0, 0))
        elif mode == "goal":
            text = font.render("Click to place GOAL (red)", True, (0, 0, 0))
        elif mode == "dynamic":
            text = font.render("LEFT: Draw dynamic | RIGHT: Delete obstacles | S: Save | Ctrl+Z: Undo | E: Edit static", True, (0, 0, 0))
        elif mode == "static_edit":
            text = font.render("LEFT: Add static | RIGHT: Delete static | D: Dynamic mode | S: Save", True, (0, 0, 0))
        else:
            text = font.render("Press S to save, R to reset", True, (0, 0, 0))
        
        screen.blit(text, (ENV_PADDING, ENV_PADDING + ENV_PIXEL + 10))
        
        # Hiển thị thông tin thêm
        if mode in ("dynamic", "static_edit"):
            info_text = font.render(f"Static: {len(static_obstacles)} | Dynamic: {len(dynamic_obstacles)}", True, (0, 0, 0))
            screen.blit(info_text, (ENV_PADDING, ENV_PADDING + ENV_PIXEL + 30))
        
        return True
    
    finished = False
    while not finished:
        draw_all()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None, None
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                px, py = event.pos
                
                # Kiểm tra click trong vùng map
                if (ENV_PADDING <= px <= ENV_PADDING + ENV_PIXEL and 
                    ENV_PADDING <= py <= ENV_PADDING + ENV_PIXEL):
                    
                    if mode == "start":
                        # Snap to grid
                        grid_x = ((px - ENV_PADDING) // CELL_SIZE) * CELL_SIZE + ENV_PADDING + CELL_SIZE // 2
                        grid_y = ((py - ENV_PADDING) // CELL_SIZE) * CELL_SIZE + ENV_PADDING + CELL_SIZE // 2
                        start_pos = (grid_x, grid_y)
                        mode = "goal"
                    elif mode == "goal":
                        # Snap to grid
                        grid_x = ((px - ENV_PADDING) // CELL_SIZE) * CELL_SIZE + ENV_PADDING + CELL_SIZE // 2
                        grid_y = ((py - ENV_PADDING) // CELL_SIZE) * CELL_SIZE + ENV_PADDING + CELL_SIZE // 2
                        goal_pos = (grid_x, grid_y)
                        mode = "dynamic"
                    elif mode == "dynamic":
                        if event.button == 1:  # Left click - vẽ dynamic obstacle
                            drawing = True
                            mx, my = px, py
                        elif event.button == 3:  # Right click - xóa obstacles
                            # Thử xóa static obstacle trước
                            obstacle_index = find_obstacle_at_position(px, py, static_obstacles)
                            if obstacle_index is not None:
                                static_obstacles.pop(obstacle_index)
                                # Cập nhật binary_map
                                grid_i = (static_obstacles[obstacle_index].y - ENV_PADDING) // CELL_SIZE
                                grid_j = (static_obstacles[obstacle_index].x - ENV_PADDING) // CELL_SIZE
                                if 0 <= grid_i < binary_map.shape[0] and 0 <= grid_j < binary_map.shape[1]:
                                    binary_map[int(grid_i)][int(grid_j)] = 0
                                print(f"Đã xóa static obstacle tại vị trí ({px}, {py})")
                            else:
                                # Nếu không có static obstacle, thử xóa dynamic obstacle
                                obstacle_index = find_obstacle_at_position(px, py, dynamic_obstacles)
                                if obstacle_index is not None:
                                    dynamic_obstacles.pop(obstacle_index)
                                    print(f"Đã xóa dynamic obstacle tại vị trí ({px}, {py})")
                    elif mode == "static_edit":
                        grid_j = (px - ENV_PADDING) // CELL_SIZE
                        grid_i = (py - ENV_PADDING) // CELL_SIZE
                        if event.button == 1:  # Left click - thêm static obstacle
                            # Kiểm tra đã có obstacle chưa
                            if 0 <= grid_i < binary_map.shape[0] and 0 <= grid_j < binary_map.shape[1]:
                                if binary_map[grid_i][grid_j] == 0:
                                    binary_map[grid_i][grid_j] = 1
                                    x = ENV_PADDING + grid_j * CELL_SIZE + CELL_SIZE / 2
                                    y = ENV_PADDING + grid_i * CELL_SIZE + CELL_SIZE / 2
                                    static_obstacles.append(Obstacle(x, y, CELL_SIZE, CELL_SIZE, True, [0, 0]))
                        elif event.button == 3:  # Right click - xóa static obstacle
                            obstacle_index = find_obstacle_at_position(px, py, static_obstacles)
                            if obstacle_index is not None:
                                obs = static_obstacles.pop(obstacle_index)
                                grid_i = (obs.y - ENV_PADDING) // CELL_SIZE
                                grid_j = (obs.x - ENV_PADDING) // CELL_SIZE
                                if 0 <= grid_i < binary_map.shape[0] and 0 <= grid_j < binary_map.shape[1]:
                                    binary_map[int(grid_i)][int(grid_j)] = 0
                                print(f"Đã xóa static obstacle tại vị trí ({px}, {py})")
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if drawing and mode == "dynamic" and event.button == 1:  # Chỉ xử lý left click
                    # Kết thúc vẽ dynamic obstacle
                    x2, y2 = event.pos
                    width = abs(x2 - mx)
                    height = abs(y2 - my)
                    
                    # Chỉ tạo obstacle nếu có kích thước đủ lớn
                    if width > 5 and height > 5:
                        center_x = (mx + x2) / 2
                        center_y = (my + y2) / 2
                        # Tạo velocity ngẫu nhiên cho dynamic obstacle
                        velocity = np.random.randn(2) * 2
                        dynamic_obstacles.append(Obstacle(center_x, center_y, width, height, False, velocity))
                    
                    drawing = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  # Save
                    if mode in ("dynamic", "static_edit") and start_pos and goal_pos:
                        finished = True
                    elif mode == "done":
                        finished = True
                elif event.key == pygame.K_r:  # Reset
                    start_pos = None
                    goal_pos = None
                    dynamic_obstacles = []
                    static_obstacles = binary_to_obstacles(binary_map)
                    mode = "start"
                elif event.key == pygame.K_z and pygame.key.get_mods() & pygame.KMOD_CTRL:  # Undo
                    if mode == "dynamic" and dynamic_obstacles:
                        dynamic_obstacles.pop()
                elif event.key == pygame.K_e and mode == "dynamic":
                    mode = "static_edit"
                elif event.key == pygame.K_d and mode == "static_edit":
                    mode = "dynamic"
        
        pygame.display.update()
    
    pygame.quit()
    if start_pos is None or goal_pos is None:
        return None, None, None
    return start_pos, goal_pos, static_obstacles + dynamic_obstacles

def main():
    image_path = input("Nhập đường dẫn đến file ảnh PNG: ")
    
    if not os.path.exists(image_path):
        print(f"File không tồn tại: {image_path}")
        return
    
    # Tải ảnh và chuyển thành binary map
    print("Đang xử lý ảnh...")
    binary_map = load_image_to_binary(image_path, TARGET_SIZE)
    
    if binary_map is None:
        print("Không thể xử lý ảnh")
        return
    
    print(f"Đã chuyển ảnh thành binary map {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Số obstacles: {np.sum(binary_map)}")
    
    # Chọn start, goal và vẽ dynamic obstacles
    print("Đang mở giao diện để chọn start, goal và vẽ dynamic obstacles...")
    start_pos, goal_pos, obstacles = select_start_goal_and_dynamic_obstacles(binary_map)
    
    if start_pos is None or goal_pos is None or obstacles is None:
        print("Chưa chọn start hoặc goal hoặc đã hủy")
        return
    
    # Lấy tên map từ file
    map_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Lưu file
    output_file = os.path.join(get_project_root(), f"MapData_{map_name}.py")
    with open(output_file, 'w') as f:
        f.write('from environment.Obstacle import Obstacle\n\n')
        f.write(f"{map_name} = {{\n")
        f.write(f"    'Start': {start_pos},\n")
        f.write(f"    'Goal': {goal_pos},\n")
        f.write("    'Obstacles': [\n")
        for obs in obstacles:
            f.write(f"        {obs.__str__()},\n")
        f.write("    ]\n}\n")
    
    print(f"Đã lưu map '{map_name}' vào {output_file}")
    print(f"Start: {start_pos}")
    print(f"Goal: {goal_pos}")
    print(f"Số obstacles: {len(obstacles)}")

if __name__ == "__main__":
    main() 