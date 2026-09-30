use std::cell::RefCell;
use std::collections::VecDeque;

use lights::{Image, LightsError};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{ClassType, DeclaredClass, declare_class, msg_send_id, mutability};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSBackingStoreType, NSEvent, NSEventMask,
    NSResponder, NSView, NSWindow, NSWindowDelegate, NSWindowStyleMask,
};
use objc2_foundation::{
    CGSize, MainThreadMarker, NSDate, NSDefaultRunLoopMode, NSNotification, NSObject,
    NSObjectProtocol, NSPoint, NSRect, NSSize, NSString,
};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandEncoder, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLDrawable,
    MTLOrigin, MTLPixelFormat, MTLResourceOptions, MTLSize, MTLTexture,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

pub const EVENTS_PER_PUMP: usize = 4096;
pub const BYTES_PER_PRESENTED_PIXEL: usize = 4;

#[derive(Clone, Debug, PartialEq)]
pub enum HostEvent {
    PointerPressed { x: f64, y: f64 },
    PointerMoved { x: f64, y: f64 },
    PointerReleased { x: f64, y: f64 },
    KeyPressed { code: u16, characters: String },
    Resized { width: u32, height: u32, scale: f64 },
    CloseRequested,
}

#[derive(Debug, thiserror::Error)]
pub enum HostError {
    #[error("this Mac offers no Metal device")]
    NoMetalDevice,
    #[error("the Metal device gave no command queue")]
    NoCommandQueue,
    #[error("the Metal device could not allocate a shared buffer of {0} bytes")]
    NoBuffer(usize),
    #[error("the layer vended no drawable")]
    NoDrawable,
    #[error("the command queue gave no command buffer or no blit encoder")]
    NoEncoder,
    #[error("a {width}x{height} frame carries {pixels} pixels")]
    FrameLength { width: u32, height: u32, pixels: usize },
    #[error("the frame is {frame_width}x{frame_height} and its drawable {drawable_width}x{drawable_height}")]
    DrawableSize {
        frame_width: usize,
        frame_height: usize,
        drawable_width: usize,
        drawable_height: usize,
    },
    #[error("the command buffer ended with Metal status {0}")]
    CommandBuffer(usize),
    #[error("a {width}x{height} frame at a row stride of {stride} bytes does not fit {len} bytes")]
    Target {
        width: u32,
        height: u32,
        stride: usize,
        len: usize,
    },
    #[error("lights could not fill the frame: {0}")]
    Fill(#[from] LightsError),
}

struct ViewState {
    events: RefCell<VecDeque<HostEvent>>,
}

declare_class!(
    struct HostView;

    unsafe impl ClassType for HostView {
        #[inherits(NSResponder, NSObject)]
        type Super = NSView;
        type Mutability = mutability::MainThreadOnly;
        const NAME: &'static str = "LightsHostView";
    }

    impl DeclaredClass for HostView {
        type Ivars = ViewState;
    }

    unsafe impl HostView {
        #[method(isFlipped)]
        fn is_flipped(&self) -> bool {
            true
        }

        #[method(acceptsFirstResponder)]
        fn accepts_first_responder(&self) -> bool {
            true
        }

        #[method(acceptsFirstMouse:)]
        fn accepts_first_mouse(&self, _event: Option<&NSEvent>) -> bool {
            true
        }

        #[method(mouseDown:)]
        fn mouse_down(&self, event: &NSEvent) {
            let (x, y) = self.pixel_position(event);
            self.push(HostEvent::PointerPressed { x, y });
        }

        #[method(mouseDragged:)]
        fn mouse_dragged(&self, event: &NSEvent) {
            let (x, y) = self.pixel_position(event);
            self.push(HostEvent::PointerMoved { x, y });
        }

        #[method(mouseMoved:)]
        fn mouse_moved(&self, event: &NSEvent) {
            let (x, y) = self.pixel_position(event);
            self.push(HostEvent::PointerMoved { x, y });
        }

        #[method(mouseUp:)]
        fn mouse_up(&self, event: &NSEvent) {
            let (x, y) = self.pixel_position(event);
            self.push(HostEvent::PointerReleased { x, y });
        }

        #[method(keyDown:)]
        fn key_down(&self, event: &NSEvent) {
            let code = unsafe { event.keyCode() };
            let characters = unsafe { event.charactersIgnoringModifiers() }
                .map(|s| s.to_string())
                .unwrap_or_default();
            self.push(HostEvent::KeyPressed { code, characters });
        }
    }

    unsafe impl NSObjectProtocol for HostView {}

    unsafe impl NSWindowDelegate for HostView {
        #[method(windowShouldClose:)]
        fn window_should_close(&self, _sender: &NSWindow) -> bool {
            self.push(HostEvent::CloseRequested);
            false
        }

        #[method(windowDidResize:)]
        fn window_did_resize(&self, _notification: &NSNotification) {
            self.push_resized();
        }

        #[method(windowDidChangeBackingProperties:)]
        fn window_did_change_backing_properties(&self, _notification: &NSNotification) {
            self.push_resized();
        }
    }
);

impl HostView {
    fn new(mtm: MainThreadMarker, frame: NSRect) -> Retained<Self> {
        let this = mtm.alloc().set_ivars(ViewState {
            events: RefCell::new(VecDeque::new()),
        });
        unsafe { msg_send_id![super(this), initWithFrame: frame] }
    }

    fn scale(&self) -> f64 {
        self.window().map_or(1.0, |w| w.backingScaleFactor())
    }

    fn pixel_position(&self, event: &NSEvent) -> (f64, f64) {
        let point = self.convertPoint_fromView(unsafe { event.locationInWindow() }, None);
        let scale = self.scale();
        (point.x * scale, point.y * scale)
    }

    fn pixel_size(&self) -> (u32, u32) {
        let bounds = self.bounds();
        let scale = self.scale();
        (
            (bounds.size.width * scale).round() as u32,
            (bounds.size.height * scale).round() as u32,
        )
    }

    fn push(&self, event: HostEvent) {
        self.ivars().events.borrow_mut().push_back(event);
    }

    fn push_resized(&self) {
        if let Some(layer) = unsafe { self.layer() } {
            layer.setContentsScale(self.scale());
        }
        let (width, height) = self.pixel_size();
        self.push(HostEvent::Resized {
            width,
            height,
            scale: self.scale(),
        });
    }
}

pub struct Host {
    app: Retained<NSApplication>,
    window: Retained<NSWindow>,
    view: Retained<HostView>,
    layer: Retained<CAMetalLayer>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    staging: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    readback: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl Host {
    pub fn open(
        mtm: MainThreadMarker,
        title: &str,
        width: f64,
        height: f64,
        policy: NSApplicationActivationPolicy,
    ) -> Result<Self, HostError> {
        let app = NSApplication::sharedApplication(mtm);
        app.setActivationPolicy(policy);
        unsafe { app.finishLaunching() };
        let device = unsafe { Retained::from_raw(MTLCreateSystemDefaultDevice()) }
            .ok_or(HostError::NoMetalDevice)?;
        let queue = device.newCommandQueue().ok_or(HostError::NoCommandQueue)?;
        let content = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(width, height));
        let style = NSWindowStyleMask::Titled
            | NSWindowStyleMask::Closable
            | NSWindowStyleMask::Miniaturizable
            | NSWindowStyleMask::Resizable;
        let window = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                mtm.alloc(),
                content,
                style,
                NSBackingStoreType::NSBackingStoreBuffered,
                false,
            )
        };
        unsafe { window.setReleasedWhenClosed(false) };
        window.setTitle(&NSString::from_str(title));
        let view = HostView::new(mtm, content);
        let layer = unsafe { CAMetalLayer::new() };
        unsafe {
            layer.setDevice(Some(&device));
            layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            layer.setFramebufferOnly(false);
            view.setLayer(Some(&layer));
        }
        layer.setContentsScale(window.backingScaleFactor());
        view.setWantsLayer(true);
        window.setContentView(Some(&view));
        window.setDelegate(Some(ProtocolObject::from_ref(&*view)));
        window.setAcceptsMouseMovedEvents(true);
        window.makeFirstResponder(Some(&view));
        window.center();
        window.makeKeyAndOrderFront(None);
        Ok(Self {
            app,
            window,
            view,
            layer,
            device,
            queue,
            staging: None,
            readback: None,
        })
    }

    pub fn app(&self) -> &NSApplication {
        &self.app
    }

    pub fn window(&self) -> &NSWindow {
        &self.window
    }

    pub fn scale_factor(&self) -> f64 {
        self.view.scale()
    }

    pub fn pixel_size(&self) -> (u32, u32) {
        self.view.pixel_size()
    }

    pub fn pump(&self, wait_seconds: f64) -> Vec<HostEvent> {
        let mode = unsafe { NSDefaultRunLoopMode };
        let mut until = unsafe { NSDate::dateWithTimeIntervalSinceNow(wait_seconds) };
        for _ in 0..EVENTS_PER_PUMP {
            let next = unsafe {
                self.app
                    .nextEventMatchingMask_untilDate_inMode_dequeue(NSEventMask::Any, Some(&until), mode, true)
            };
            let Some(event) = next else {
                break;
            };
            unsafe { self.app.sendEvent(&event) };
            until = unsafe { NSDate::distantPast() };
        }
        unsafe { self.app.updateWindows() };
        self.view.ivars().events.borrow_mut().drain(..).collect()
    }

    pub fn present(&mut self, frame: &Image) -> Result<(), HostError> {
        self.encode(frame.width, frame.height, |bytes, stride| write_bgra(frame, bytes, stride), false)
            .map(|_| ())
    }

    pub fn present_filled<F>(&mut self, width: u32, height: u32, fill: F) -> Result<(), HostError>
    where
        F: FnOnce(&mut [u8], usize) -> Result<(), HostError>,
    {
        self.encode(width, height, fill, false).map(|_| ())
    }

    pub fn present_and_read_back<F>(&mut self, width: u32, height: u32, fill: F) -> Result<Vec<u8>, HostError>
    where
        F: FnOnce(&mut [u8], usize) -> Result<(), HostError>,
    {
        self.encode(width, height, fill, true).map(Option::unwrap_or_default)
    }

    fn encode<F>(&mut self, frame_width: u32, frame_height: u32, fill: F, read_back: bool) -> Result<Option<Vec<u8>>, HostError>
    where
        F: FnOnce(&mut [u8], usize) -> Result<(), HostError>,
    {
        let (width, height) = (frame_width as usize, frame_height as usize);
        if width == 0 || height == 0 {
            return Err(HostError::FrameLength {
                width: frame_width,
                height: frame_height,
                pixels: 0,
            });
        }
        let row = width * BYTES_PER_PRESENTED_PIXEL;
        let length = row * height;
        let staging = shared_buffer(&self.device, &mut self.staging, length)?;
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(staging.contents().as_ptr().cast::<u8>(), length) };
        fill(bytes, row)?;
        let size = CGSize::new(width as f64, height as f64);
        unsafe {
            if self.layer.drawableSize() != size {
                self.layer.setDrawableSize(size);
            }
        }
        let drawable = unsafe { self.layer.nextDrawable() }.ok_or(HostError::NoDrawable)?;
        let texture = unsafe { drawable.texture() };
        if texture.width() != width || texture.height() != height {
            return Err(HostError::DrawableSize {
                frame_width: width,
                frame_height: height,
                drawable_width: texture.width(),
                drawable_height: texture.height(),
            });
        }
        let commands = self.queue.commandBuffer().ok_or(HostError::NoEncoder)?;
        let blit = commands.blitCommandEncoder().ok_or(HostError::NoEncoder)?;
        let extent = MTLSize {
            width,
            height,
            depth: 1,
        };
        let origin = MTLOrigin { x: 0, y: 0, z: 0 };
        unsafe {
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &staging, 0, row, length, extent, &texture, 0, 0, origin,
            );
        }
        let target = if read_back {
            let target = shared_buffer(&self.device, &mut self.readback, length)?;
            unsafe {
                blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                    &texture, 0, 0, origin, extent, &target, 0, row, length,
                );
            }
            Some(target)
        } else {
            None
        };
        blit.endEncoding();
        let shown: &ProtocolObject<dyn MTLDrawable> = ProtocolObject::from_ref(&*drawable);
        commands.presentDrawable(shown);
        commands.commit();
        unsafe { commands.waitUntilCompleted() };
        let status = commands.status();
        if status != MTLCommandBufferStatus::Completed {
            return Err(HostError::CommandBuffer(status.0));
        }
        Ok(target.map(|b| {
            unsafe { std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>(), length) }.to_vec()
        }))
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        self.window.setDelegate(None);
        self.window.close();
    }
}

pub fn write_bgra(frame: &Image, bytes: &mut [u8], stride: usize) -> Result<(), HostError> {
    let (width, height) = (frame.width as usize, frame.height as usize);
    if width == 0 || height == 0 || frame.pixels.len() != width * height {
        return Err(HostError::FrameLength {
            width: frame.width,
            height: frame.height,
            pixels: frame.pixels.len(),
        });
    }
    let row = width * BYTES_PER_PRESENTED_PIXEL;
    if stride < row || bytes.len() < stride * (height - 1) + row {
        return Err(HostError::Target {
            width: frame.width,
            height: frame.height,
            stride,
            len: bytes.len(),
        });
    }
    for (line, pixels) in bytes.chunks_mut(stride).zip(frame.pixels.chunks_exact(width)) {
        for (out, pixel) in line.chunks_exact_mut(BYTES_PER_PRESENTED_PIXEL).zip(pixels) {
            out.copy_from_slice(&[pixel[2], pixel[1], pixel[0], u8::MAX]);
        }
    }
    Ok(())
}

fn shared_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    slot: &mut Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    length: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, HostError> {
    if let Some(buffer) = slot.as_ref().filter(|b| b.length() == length) {
        return Ok(buffer.clone());
    }
    let buffer = device
        .newBufferWithLength_options(length, MTLResourceOptions::MTLResourceStorageModeShared)
        .ok_or(HostError::NoBuffer(length))?;
    *slot = Some(buffer.clone());
    Ok(buffer)
}
