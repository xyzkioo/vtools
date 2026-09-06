# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import colors


class SecurityAlarm(BaseSolution):
    """管理实时监控安全报警功能的类。

    此类继承 BaseSolution，可监控帧中的对象，在检测总数超过指定阈值时发送电子邮件通知，
    并为输出帧添加可视化标注。

    属性：
        email_sent (bool): 标记当前事件是否已经发送过邮件。
        records (int): 触发报警所需的检测对象数量阈值。
        server (smtplib.SMTP): 用于发送报警邮件的 SMTP 服务器连接。
        to_email (str): 报警邮件收件人地址。
        from_email (str): 报警邮件发件人地址。

    方法：
        authenticate: 设置用于发送报警邮件的服务器身份验证。
        send_email: 发送包含检测详情和图像附件的报警邮件。
        process: 监控帧、处理检测结果，并在达到阈值时触发报警。

    示例：
        >>> security = SecurityAlarm()
        >>> security.authenticate("abc@gmail.com", "1111222233334444", "xyz@gmail.com")
        >>> frame = cv2.imread("frame.jpg")
        >>> results = security.process(frame)
    """

    def __init__(self, **kwargs: Any) -> None:
        """使用实时对象监控参数初始化 SecurityAlarm 类。

        参数：
            **kwargs (Any): 传递给父类的其他关键字参数。
        """
        super().__init__(**kwargs)
        self.email_sent = False
        self.records = self.CFG["records"]
        self.server = None
        self.to_email = ""
        self.from_email = ""

    def authenticate(self, from_email: str, password: str, to_email: str) -> None:
        """验证用于发送报警通知的电子邮件服务器。

        此方法会与 SMTP 服务器建立安全连接，并使用提供的凭据登录。

        参数：
            from_email (str): 发件人电子邮件地址。
            password (str): 发件人电子邮件账户密码。
            to_email (str): 收件人电子邮件地址。

        示例：
            >>> alarm = SecurityAlarm()
            >>> alarm.authenticate("sender@example.com", "password123", "recipient@example.com")
        """
        import smtplib

        self.server = smtplib.SMTP("smtp.gmail.com", 587)
        self.server.starttls()
        self.server.login(from_email, password)
        self.to_email = to_email
        self.from_email = from_email

    def send_email(self, im0, records: int = 5) -> None:
        """发送包含图像附件的电子邮件通知，说明检测到的对象数量。

        此方法会编码输入图像，组合包含检测详情的邮件，并将邮件发送给指定收件人。

        参数：
            im0 (np.ndarray): 要作为附件发送的输入图像或帧。
            records (int, 可选): 要包含在邮件中的检测对象数量。

        示例：
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> alarm.send_email(frame, records=10)
        """
        from email.mime.image import MIMEImage
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        import cv2

        img_bytes = cv2.imencode(".jpg", im0)[1].tobytes()  # 将图像编码为 JPEG

        # 创建邮件
        message = MIMEMultipart()
        message["From"] = self.from_email
        message["To"] = self.to_email
        message["Subject"] = "Security Alert"

        # 添加文本邮件正文
        message_body = f"Ultralytics alert: {records} object(s) detected."
        message.attach(MIMEText(message_body))

        # 添加图像附件
        image_attachment = MIMEImage(img_bytes, name="ultralytics.jpg")
        message.attach(image_attachment)

        # 发送邮件
        try:
            self.server.send_message(message)
            LOGGER.info("Email sent successfully!")
        except Exception as e:
            LOGGER.error(f"Failed to send email: {e}")

    def process(self, im0) -> SolutionResults:
        """监控帧并处理对象检测结果，在达到阈值时触发报警。

        此方法处理输入帧，提取检测结果并使用边界框标注帧；当检测对象数量达到或超过指定阈值，
        且尚未发送过报警时，发送电子邮件通知。

        参数：
            im0 (np.ndarray): 要处理和标注的输入图像或帧。

        返回：
            (SolutionResults): 包含处理后图像 `plot_im`、`total_tracks`（跟踪对象总数）和
                `email_sent`（是否触发电子邮件报警）。

        示例：
            >>> alarm = SecurityAlarm()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> results = alarm.process(frame)
        """
        self.extract_tracks(im0)  # 提取跟踪结果
        annotator = SolutionAnnotator(im0, line_width=self.line_width)  # 初始化标注器

        # 遍历边界框和类别索引
        for box, cls in zip(self.boxes, self.clss):
            # 绘制边界框
            annotator.box_label(box, label=self.names[cls], color=colors(cls, True))

        total_det = len(self.clss)
        if total_det >= self.records and not self.email_sent and self.server:
            self.send_email(im0, total_det)
            self.email_sent = True

        plot_im = annotator.result()
        self.display_output(plot_im)  # 使用基类方法显示输出

        # 返回 SolutionResults
        return SolutionResults(plot_im=plot_im, total_tracks=len(self.track_ids), email_sent=self.email_sent)
