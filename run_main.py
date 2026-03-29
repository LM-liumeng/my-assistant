# import os
# # 临时启用自动确认
# os.environ['AUTO_CONFIRM'] = '1'
#
# from app.main import create_app
#
# if __name__ == '__main__':
#     app = create_app()
#     # 如有需要调整端口
#     app.run(host='127.0.0.1', port=5000)


import os
import smtplib
import imaplib

# 要检查的环境变量列表
required_vars = [
    'SMTP_SERVER', 'SMTP_PORT', 'SMTP_USERNAME', 'SMTP_PASSWORD',
    'IMAP_SERVER', 'IMAP_PORT', 'IMAP_USERNAME', 'IMAP_PASSWORD'
]

# 检查缺失变量
missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    print(f"缺失的环境变量: {', '.join(missing_vars)}")
    print("请确保所有变量已正确设置。")
else:
    print("所有环境变量已设置。开始连接测试...")

    # 测试 SMTP 连接
    try:
        smtp_server = os.environ['SMTP_SERVER']
        smtp_port = int(os.environ['SMTP_PORT'])
        smtp_user = os.environ['SMTP_USERNAME']
        smtp_pass = os.environ['SMTP_PASSWORD']

        with smtplib.SMTP(smtp_server, smtp_port) as smtp:
            smtp.starttls()  # 启用 TLS（如果端口支持）
            smtp.login(smtp_user, smtp_pass)
        print("SMTP 连接和认证成功！")
    except Exception as e:
        print(f"SMTP 连接失败: {str(e)}")

    # 测试 IMAP 连接
    try:
        imap_server = os.environ['IMAP_SERVER']
        imap_port = int(os.environ['IMAP_PORT'])
        imap_user = os.environ['IMAP_USERNAME']
        imap_pass = os.environ['IMAP_PASSWORD']

        with imaplib.IMAP4_SSL(imap_server, imap_port) as imap:
            imap.login(imap_user, imap_pass)
        print("IMAP 连接和认证成功！")
    except Exception as e:
        print(f"IMAP 连接失败: {str(e)}")