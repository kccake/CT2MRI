import colorama
colorama.init()

class BugFree:
    def __init__(self, config):
        self.img_name = config.get('img_name', 'buddha')
        self.start_color = config.get('start_color', 'FF7EC7')
        self.end_color = config.get('end_color', 'FFED46')
        self.bugfree_img_path = config.get('bugfree_img_path', './utils/bugfree_img/')
        self.bug_free_texts = self.get_bug_free_gradient_texts()
    
    def get_bug_free_str(self):
        try:
            with open(self.bugfree_img_path + self.img_name, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return "上天保佑，代码无BUG!\n" * 3

    def get_gradient_text(self, text):
        def hex2RGB(hex):
            return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

        def interpolate(start, end, factor):
            return int(start + (end - start) * factor)
        
        def get_color_code(r, g, b):
            return f'\033[38;2;{r};{g};{b}m'
        
        # 将颜色转换为RGB值
        start_r, start_g, start_b = hex2RGB(self.start_color)
        end_r, end_g, end_b = hex2RGB(self.end_color)

        gradient_text = ""

        for i, char in enumerate(text):
            factor = i / (len(text) - 1) if len(text) > 1 else 0
            r = interpolate(start_r, end_r, factor)
            g = interpolate(start_g, end_g, factor)
            b = interpolate(start_b, end_b, factor)

            # 生成渐变色字符
            color_code = get_color_code(r, g, b)
            gradient_text += f"{color_code}{char}"
        
        # 重置颜色
        gradient_text += colorama.Style.RESET_ALL
        return gradient_text

    def get_gradient_texts(self, texts):
        max_length = max([len(text) for text in texts])
        gradient_texts = []
        for text in texts:
            gradient_texts.append(self.get_gradient_text(text.ljust(max_length)))
        return gradient_texts

    def get_bug_free_gradient_texts(self):
        bug_free_texts = self.get_bug_free_str().splitlines()  # 按行分割
        gt = self.get_gradient_texts(bug_free_texts)
        return gt
    
    def print_bug_free_gradient_texts(self):
        for gt in self.bug_free_texts:
            print(gt)
    
    def __call__(self):
        self.print_bug_free_gradient_texts()
