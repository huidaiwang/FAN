import ollama

# 流式输出
def api_generate(text: str):
    print(f'提问：{text}')

    stream = ollama.generate(
        stream=True,
        model='fan1.0',  # 确保模型名称是正确的
        prompt=text,
    )

    print('-----------------------------------------')
    for chunk in stream:
        if not chunk['done']:
            print(chunk['response'], end='', flush=True)
        else:
            print('\n')
            print('-----------------------------------------')
            print(f'总耗时：{chunk["total_duration"]}')
            print('-----------------------------------------')

if __name__ == '__main__':
    # 流式输出
    api_generate(text='天空为什么是蓝色的？')

    # 非流式输出
    # content = ollama.generate(model='fan1.0', prompt='天空为什么是蓝色的？')
    # print(content)
