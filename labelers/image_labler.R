require(shiny)
require(shinyFiles)
require(shinyjs)
require(base64enc)

# set the number of images after which to save the data
n_save <- 10
# TODO rename file to stop overwriting
csv_file <- 'test_labels.csv'

classes <- list("1: Clear" = 1,
                "2: Dark" = 2,
                "3: Foggy" = 3)

# UI definition
ui <- fluidPage(
  useShinyjs(),
  titlePanel("Image Labeler"),
  sidebarLayout(
    sidebarPanel(
      shinyDirButton('dir', 'Select Image Directory', 'Select the directory containing images'),
      actionButton("start_unlabeled", "Label Unlabeled"),
      actionButton("start_all", "Label All"),
      actionButton("quit_btn", "Close App"),
      hr(),
      uiOutput("image_ui"),
      hr(),
      radioButtons("class_input", "Select Class:", 
                   choices = classes),
      textInput("comment_input", "Add Comment:", ""),
      actionButton("prev_btn", "Previous"),
      actionButton("next_btn", "Next"),
      width = 3
    ),
    mainPanel(
      uiOutput("image_display"),
      hr(),
      verbatimTextOutput("status"),
      tags$script(HTML("
        $(document).on('keydown', function (e) {
          Shiny.onInputChange('keypress', e.which);
        });
      "))
    )
  )
)

server <- function(input, output, session) {
  roots <- c(wd = '..')
  shinyDirChoose(input, 'dir', roots = roots)
  
  # dt to store state
  values <- reactiveValues(
    images = NULL,
    current_index = 1,
    labels = data.frame(image = character(),
                        class = integer(),
                        comment = character(),
                        stringsAsFactors = FALSE),
    selected_dir = NULL,
    class_input = NULL
  )
  
  # function to initialize labeling
  initializeLabeling <- function(mode) {
    req(input$dir)
    # get the selected directory
    path <- parseDirPath(roots, input$dir)
    if (length(path) == 0) {
      showNotification("No directory selected.", type = "error")
      return()
    }
    directory <- path
    # get list of image files
    image_extensions <- c("jpg", "jpeg", "png", "bmp", "gif")
    files <- list.files(directory, pattern = paste0("\\.(", paste(image_extensions, collapse = "|"), ")$"), full.names = TRUE)
    if (length(files) == 0) {
      showNotification("No images found in the selected directory.", type = "error")
      return()
    }
    # load existing labels if any
    if (file.exists(csv_file)) {
      values$labels <- read.csv(csv_file, stringsAsFactors = FALSE)
    } else {
      values$labels <- data.frame(image = character(),
                                  class = integer(),
                                  comment = character(),
                                  stringsAsFactors = FALSE)
    }
    if (mode == "unlabeled") {
      # start from the first unlabeled image
      labeled_images <- values$labels$image
      unlabeled_images <- setdiff(basename(files), labeled_images)
      if (length(unlabeled_images) == 0) {
        showNotification("All images are already labeled.", type = "message")
        return()
      }
      # subset images to unlabeled images
      files <- files[basename(files) %in% unlabeled_images]
      values$current_index <- 1
    } else if (mode == "all") {
      # reset labels
      values$labels <- data.frame(image = character(),
                                  class = integer(),
                                  comment = character(),
                                  stringsAsFactors = FALSE)
      values$current_index <- 1
    }
    values$images <- files
    output$status <- renderText({
      paste("Loaded", length(values$images), "images.")
    })
    # display the first image
    updateImage()
  }
  
  # observe "Label Unlabeled" button
  observeEvent(input$start_unlabeled, {
    initializeLabeling("unlabeled")
  })
  
  # observe "Label All" button
  observeEvent(input$start_all, {
    initializeLabeling("all")
  })
  
  # close button handler
  observeEvent(input$quit_btn, {
    # save labels
    if (nrow(values$labels) > 0) {
      write.csv(values$labels, csv_file, row.names = FALSE)
    }
    stopApp()  # close the app
  })
  
  # function to update the displayed image
  updateImage <- function() {
    req(values$images)
    img_path <- values$images[values$current_index]
    output$image_display <- renderUI({
      tags$img(src = base64enc::dataURI(file = img_path, mime = "image/png"), width = "100%")
    })
    output$image_ui <- renderUI({
      tags$h4(basename(img_path))
    })
    # update class and comment inputs if previously saved
    existing_label <- values$labels[values$labels$image == basename(img_path), ]
    if (nrow(existing_label) > 0) {
      updateRadioButtons(session, "class_input", selected = existing_label$class)
      updateTextInput(session, "comment_input", value = existing_label$comment)
    } else {
      updateRadioButtons(session, "class_input", selected = character(0))
      updateTextInput(session, "comment_input", value = "")
    }
    # display the selected class
    output$status <- renderText({
      paste(sprintf("Press 1-%s to select class, Enter/Space to confirm, Backspace to go back.", length(classes)))
    })
  }
  
  # class selection via buttons handler
  observeEvent(input$class_input, {
    if (!is.null(input$class_input)) {
      # update values$class_input when a class is selected via radio buttons
      values$class_input <- as.integer(input$class_input)
      # move focus away from radio buttons to prevent enter key from re-triggering selection
      runjs("document.activeElement.blur();")
    }
  })
  
  # keypress handler
  observeEvent(input$keypress, {
    req(input$keypress)
    key <- input$keypress
    if (key >= 49 && key <= 49 + length(classes)) {
      values$class_input <- key - 48  # convert ascii code to number
      # update radio buttons to reflect the selection
      updateRadioButtons(session, "class_input", selected = values$class_input)
    } else if (key == 13 || key == 32) {  # enter or space
      # save current label
      saveLabel()
      if (values$current_index < length(values$images)) {
        values$current_index <- values$current_index + 1
        updateImage()
      } else {
        showNotification("This is the last image.", type = "message")
      }
    } else if (key == 8 || key == 66 || key == 98) {  # backspace or 'b' or 'B'
      if (values$current_index > 1) {
        values$current_index <- values$current_index - 1
        updateImage()
      } else {
        showNotification("This is the first image.", type = "message")
      }
    }
  })
  
  # next button handler
  observeEvent(input$next_btn, {
    req(values$images)
    # save current label
    saveLabel()
    if (values$current_index < length(values$images)) {
      values$current_index <- values$current_index + 1
      updateImage()
    } else {
      showNotification("This is the last image.", type = "message")
    }
  })
  
  # previous button handler
  observeEvent(input$prev_btn, {
    req(values$images)
    if (values$current_index > 1) {
      values$current_index <- values$current_index - 1
      updateImage()
    } else {
      showNotification("This is the first image.", type = "message")
    }
  })
  
  # function to save the current label
  saveLabel <- function() {
    img_name <- basename(values$images[values$current_index])
    class_input <- values$class_input
    comment_input <- input$comment_input
    if (is.null(class_input)) {
      showNotification("Please select a class before proceeding.", type = "warning")
      return()
    }
    # remove existing entry for this image if any
    values$labels <- values$labels[values$labels$image != img_name, ]
    # add new label
    new_label <- data.frame(image = img_name,
                            class = as.integer(class_input),
                            comment = comment_input,
                            stringsAsFactors = FALSE)
    values$labels <- rbind(values$labels, new_label)
    # save labels every n images
    if (nrow(values$labels) %% n_save == 0) {
      write.csv(values$labels, csv_file, row.names = FALSE)
      output$status <- renderText({
        paste("Labels saved to", csv_file)
      })
    }
  }
  
  # save labels at end
  session$onSessionEnded(function() {
    if (nrow(values$labels) > 0) {
      write.csv(values$labels, csv_file, row.names = FALSE)
    }
  })
}

shinyApp(ui = ui, server = server)
